import dataloader as DL
from config import config
import network as net
from math import floor, ceil
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import utils as utils
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import nvidia_smi
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

"""
ssh -L 16005:127.0.0.1:6006 sq@155.41.207.229
"""

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
        ngpu = config.n_gpu

        # network
        self.G = net.Generator(config).cuda()
        self.D = net.Discriminator(config).cuda()
        print('Generator structure: ')
        print(self.G.model)
        print('Discriminator structure: ')
        print(self.D.model)

        self.build_avgG()

        devices = [i for i in range(ngpu)]
        self.G = MyDataParallel(self.G, device_ids=devices)
        self.D = MyDataParallel(self.D, device_ids=devices)

        if config.start_resl != 2:
            self.load_model(level=config.start_resl - 1)  # if load_resl==3, model loaded was done on training at 3, should start with resl = 4

        self.nz = config.nz
        self.optimizer = config.optimizer
        self.lr = config.lr

        self.start_resl = config.start_resl
        self.max_resl = config.max_resl

        self.fadein = {'gen': None, 'dis': None}
        self.upsam_mode = self.config.G_upsam_mode  # either 'nearest' or 'trilinear'

        self.batchSize = {2: 64 * ngpu, 3: 64 * ngpu, 4: 64 * ngpu, 5: 64 * ngpu, 6: 32 * ngpu, 7: 10 * ngpu}
        self.fadeInEpochs = {2: 0, 3: 1, 4: 1, 5: 1, 6: 5000, 7: 5000}
        self.stableEpochs = {2: 0, 3: 0, 4: 0, 5: 500000, 6: 5000, 7: 5000}

        # size 16 need 5000-7000 enough
        # size 32 need 16000-30000 enough


        self.writer = SummaryWriter('runs_new')
        self.global_batch_done = 0

        # define all dataloaders into a dictionary
        self.dataloaders = {}
        for resl in range(self.start_resl, self.max_resl + 1):
            self.dataloaders[resl] = DataLoader(DL.Data(config.train_data_root + 'resl{}/'.format(2 ** resl)),
                                                     batch_size=self.batchSize[resl], shuffle=True,
                                                     drop_last=True)

        # ship new model to cuda, and update optimizer
        self.renew_everything()


    def renew_everything(self):

        self.build_avgG()

        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas,
                              weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas,
                              weight_decay=0.0)


    def train(self):

        self.test_noise = torch.randn(5, self.nz).cuda()

        for self.resl in range(self.start_resl, self.max_resl + 1):

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

            print('Generator stable structure: ')
            print(self.G.model)
            print('Discriminator stable structure: ')
            print(self.D.model)

            # stable training
            for epoch in range(self.stableEpochs[self.resl]):
                self.trainOnEpoch(self.resl, epoch, 'stable', self.stableEpochs[self.resl])

            torch.save(self.G.state_dict(), './checkpoint_dir/G_stable_{}.pth'.format(self.resl))
            torch.save(self.D.state_dict(), './checkpoint_dir/D_stable_{}.pth'.format(self.resl))


    def trainOnEpoch(self, resl, epoch, stage, maxEpochs):

        BATCH_SIZE = self.batchSize[resl]

        dataloader = self.dataloaders[resl]

        alpha = self.fadein['gen'].get_alpha() if self.fadein['gen'] else 1

        for i, (data, _) in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i
            self.global_batch_done += 1

            ###########################
            # (1) Update D network
            ###########################
            self.D.zero_grad()

            # train with real
            data = self.interpolate(data, alpha)
            real = data.cuda()
            D_real = self.D(real)
            D_real = -D_real.mean()
            D_real.backward(retain_graph=True)

            if self.config.eps_drift > 0:
                drift_loss = self.config.eps_drift * (D_real ** 2).sum()
                drift_loss.backward()

            # train with fake
            noise = torch.randn(BATCH_SIZE, self.nz).cuda()
            fake = self.G(noise).detach()
            D_fake = self.D(fake)
            D_fake = D_fake.mean()
            D_fake.backward(retain_graph=True)

            if self.config.eps_drift > 0:
                drift_loss = self.config.eps_drift * (D_fake ** 2).sum()
                drift_loss.backward()

            # train with gradient penalty
            gradient_penalty, grad_norm = self.calc_gradient_penalty(real.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake + D_real + gradient_penalty
            Wasserstein_D = - D_real - D_fake
            self.opt_d.step()

            if self.global_batch_done % self.config.ncritic == 0:
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

            # after D and G parameters updates, update the moving average of G
            for p, avg_p in zip(self.G.parameters(),
                                self.avgG.parameters()):
                avg_p.mul_(0.999).add_(0.001, p.data)

            if self.global_batch_done % 100 == 0:
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
                except:
                    pass

            if batches_done % 200 == 0:
                with torch.no_grad():
                    fake = self.G(self.test_noise).data.squeeze().cpu().numpy()
                    resolution = 2 ** (self.resl)
                    utils.save_image(fake, "./images/" + "{}^{}".format(resolution, resolution) + stage + "%d.png" % batches_done, nrow=5, ncol=3, scale=self.resl)
                    del fake
                    fake = self.avgG(self.test_noise).data.squeeze().cpu().numpy()
                    resolution = 2 ** (self.resl)
                    utils.save_image(fake, "./images/" + "{}^{}".format(resolution, resolution) + stage + "%d_avg.png" % batches_done, nrow=5, ncol=3, scale=self.resl)
                    del fake


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


    def interpolate(self, x, alpha):
        lowx = F.avg_pool3d(x, kernel_size=2, stride=2)
        lowx_upsample = nn.Upsample(scale_factor=2, mode=self.upsam_mode)(lowx)
        return x * alpha + lowx_upsample * (1-alpha)


    def load_model(self, level):
        G_pth = './checkpoint_dir/G_stable_{}.pth'.format(level)
        D_pth = './checkpoint_dir/D_stable_{}.pth'.format(level)
        for resl in range(3, level+1):
            self.G.grow_network(resl)
            self.G.flush_network()
            self.D.grow_network(resl)
            self.D.flush_network()
        self.G.load_state_dict(torch.load(G_pth))
        self.D.load_state_dict(torch.load(D_pth))
        print('successfully loaded the model at level ' + str(level))

    def build_avgG(self):
        self.avgG = deepcopy(self.G)
        for param in self.avgG.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.avgG = nn.DataParallel(self.avgG, device_ids=[0])


if __name__ == "__main__":
    trainer = ProgressiveGANTrainer(config)
    trainer.train()










