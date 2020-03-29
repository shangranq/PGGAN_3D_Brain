import dataloader as DL
from config import config
import network as net
from math import floor, ceil
import os, sys
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import utils as utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import nvidia_smi
from torch.utils.tensorboard import SummaryWriter


class MyDataParallel(nn.DataParallel):
    """
    get model attributes directly from DataParallel Wraper
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class ProgressiveGANTrainer:

    def __init__(self, config):

        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True

        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resl = 2  # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resl = config.max_resl
        self.trns_tick = config.trns_tick
        self.stab_tick = config.stab_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen': None, 'dis': None}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift

        self.batchSize = {2:64*3, 3:64*3, 4:64*3, 5:64*3, 6:32*3, 7:16*3}      # key is resl, content is batchsize for each scale
        self.fadeInEpochs = {2:0, 3:5000, 4:5000, 5:5000, 6:5000, 7:5000}
        self.stableEpochs = {2:5000, 3:5000, 4:5000, 5:5000, 6:5000, 7:5000}

        # network
        self.G = net.Generator(config).cuda()
        self.D = net.Discriminator(config).cuda()
        print('Generator structure: ')
        print(self.G.model)
        print('Discriminator structure: ')
        print(self.D.model)

        self.writer = SummaryWriter('runs')
        self.global_batch_done = 0

        self.G = MyDataParallel(self.G, device_ids=[0, 1, 2])
        self.D = MyDataParallel(self.D, device_ids=[0, 1, 2])

        # define all dataloaders into a dictionary
        self.dataloaders = {}
        for self.resl in range(2, self.max_resl + 1):
            self.dataloaders[self.resl] = DataLoader(DL.Data('/scratch2/xzhou/mri/resl{}/'.format(2 ** self.resl)),
                                                     batch_size=self.batchSize[self.resl], shuffle=True,
                                                     drop_last=True)

        # ship new model to cuda, and update optimizer
        self.renew_everything()

    def renew_everything(self):

        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr/(2**(self.resl-2)), betas=betas,
                              weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr/(2**(self.resl-2)), betas=betas,
                              weight_decay=0.0)


    def train(self):

        self.test_noise = torch.randn(5, self.nz).cuda()

        for self.resl in range(2, self.max_resl + 1):

            # fadein
            if self.fadeInEpochs[self.resl]:
                self.G.grow_network(self.resl)
                self.D.grow_network(self.resl)
                self.renew_everything()
                self.fadein['gen'] = dict(self.G.model.named_children())['fadein_block']
                self.fadein['dis'] = dict(self.D.model.named_children())['fadein_block']
                alpha_step = 1.0 / float(self.fadeInEpochs[self.resl])
                for epoch in range(self.fadeInEpochs[self.resl]):
                    self.trainOnEpoch(self.resl, epoch, 'fadein', self.fadeInEpochs[self.resl])
                    self.fadein['gen'].update_alpha(alpha_step)
                    self.fadein['dis'].update_alpha(alpha_step)

            self.G.flush_network()
            self.D.flush_network()
            self.renew_everything()

            # stable training
            for epoch in range(self.stableEpochs[self.resl]):
                self.trainOnEpoch(self.resl, epoch, 'stable', self.stableEpochs[self.resl])

            torch.save(self.G.state_dict(), './checkpoint_dir/G_stable_{}.pth'.format(self.resl))
            torch.save(self.D.state_dict(), './checkpoint_dir/D_stable_{}.pth'.format(self.resl))


    def trainOnEpoch(self, resl, epoch, stage, maxEpochs):

        BATCH_SIZE = self.batchSize[resl]

        dataloader = self.dataloaders[resl]

        for i, (data, _) in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i
            self.global_batch_done += 1

            ###########################
            # (1) Update D network
            ###########################
            self.D.zero_grad()

            # train with real
            real = data.cuda()
            D_real = self.D(real)
            D_real = -D_real.mean()
            D_real.backward()

            # train with fake
            noise = torch.randn(BATCH_SIZE, self.nz).cuda()
            fake = self.G(noise).detach()
            D_fake = self.D(fake)
            D_fake = D_fake.mean()
            D_fake.backward()

            # train with gradient penalty
            gradient_penalty, grad_norm = self.calc_gradient_penalty(real.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake + D_real + gradient_penalty
            Wasserstein_D = - D_real - D_fake
            self.opt_d.step()

            if batches_done % (10 - self.resl) == 0:
                ###########################
                # (2) Update G network
                ###########################
                self.G.zero_grad()

                noise = torch.randn(BATCH_SIZE, self.nz).cuda()
                fake = self.G(noise)
                G = self.D(fake)
                G = G.mean()
                G_cost = -G
                G_cost.backward()
                self.opt_g.step()

            if batches_done % 100 == 0:
                try:
                    print(
                        stage + " {} ".format(2**self.resl) + "[Epoch %d/%d] [Batch %d/%d] [D cost: %f] [G cost: %f] [W distance: %f] [grad norm: %f]"
                        % (epoch, maxEpochs, i, len(dataloader), D_cost.item(), G_cost.item(), Wasserstein_D.item(),
                           grad_norm.item())
                    )
                    self.writer.add_scalar('Loss/D cost', D_cost.data.cpu().numpy(), self.global_batch_done)
                    self.writer.add_scalar('Loss/G cost', -G_cost.data.cpu().numpy(), self.global_batch_done)
                    self.writer.add_scalar('W_dis', Wasserstein_D.data.cpu().numpy(), self.global_batch_done)
                    self.writer.add_scalar('GradNorm', grad_norm.data.cpu().numpy(), self.global_batch_done)
                    self.writer.add_scalar('GPU%', self.show_GPU(), self.global_batch_done)
                except:
                    pass

            if batches_done % 200 == 0:
                fake = self.G(self.test_noise).data.squeeze()
                resolution = 2 ** (self.resl)
                imagegrid = utils.save_image(fake.data.cpu().numpy(), "./images/" + "{}^{}".format(resolution, resolution) + stage + "%d.png" % batches_done, nrow=5, ncol=3, scale=self.resl)
                # self.writer.add_image("./images/" + "{} {}^{}".format(self.global_batch_done, resolution,
                #                                                       resolution) + stage + "%d" % batches_done, imagegrid, dataformats='HW')

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batchSize[self.resl], 1, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.config.Lambda
        return gradient_penalty, gradients.norm(2, dim=1).mean()

    def show_GPU(self):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        # print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
        return res.memory


if __name__ == "__main__":
    trainer = ProgressiveGANTrainer(config)
    trainer.train()










