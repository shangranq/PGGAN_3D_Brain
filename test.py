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
from metrics.msssim import MultiScaleSSIM as MSSSIM
from PGAN_trainer import MyDataParallel

class ProgressiveGANTester:

    def __init__(self, config):

        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
        ngpu = 1

        # network
        self.G = net.Generator(config).cuda()

        print('Generator structure: ')
        print(self.G.model)

        devices = [i for i in range(ngpu)]
        self.G = MyDataParallel(self.G, device_ids=devices)

        self.start_resl = config.start_resl
        self.max_resl = config.max_resl


    def load_model(self, G_pth):
        if not G_pth:
            return
        level = int(G_pth.split('_')[-2])
        for self.resl in range(3, level+1):
            self.G.grow_network(self.resl)
            self.G.flush_network()
        try:
            self.G.load_state_dict(torch.load(G_pth))
        except:
            self.G = MyDataParallel(self.G, device_ids=[0])
            self.G.load_state_dict(torch.load(G_pth))
        self.start_resl = level + 1
        print('successfully loaded the model at level ' + str(level))
        print('from ', G_pth)

    def test(self, noise):
        self.G = self.G.cuda()
        with torch.no_grad():
            fake = self.G(noise).data.squeeze().cpu().numpy()
            fake = np.clip(fake, -1, 2.5)
            resol = 2 ** self.resl
            utils.save_image(fake, "./images/" + config.model_name + "/test{}^{}.png".format(resol,resol),
                             nrow=5, ncol=3, scale=self.resl)


if __name__ == "__main__":
    path = 'checkpoint_dir/P_G/'
    model_pth = ['G_4_3500.pth',
                 'avgG_5_10000.pth',
                 'avgG_6_10500.pth',
                 'avgG_7_7000.pth']
    noise = torch.randn(5, config.nz).cuda()
    for pth in model_pth:
        tester = ProgressiveGANTester(config)
        tester.load_model(path+pth)
        tester.test(noise)

