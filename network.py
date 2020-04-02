import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import copy
from config import config
from custom_layers import *


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_norm_latent = config.flag_norm_latent
        self.upsam_mode = config.upsam_mode  # either 'nearest' or 'trilinear'
        self.nc = config.nc    # color channel number of image
        self.nz = config.nz    # random noise dimension
        self.ngf = config.ngf  # first conv layer out_channel number
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_gen()


    def get_init_gen(self):
        model = nn.Sequential()
        first_block, ndim = self.first_block()
        model.add_module('first_block', first_block)
        model.add_module('to_rgb_block', self.to_rgb_block(ndim))
        self.module_names = get_module_names(model)
        return model


    def first_block(self):
        layers = []
        ndim = self.ngf
        if self.flag_norm_latent:
            layers.append(pixelwise_norm_layer())
        layers = Trans_conv(layers, self.nz, ndim, 4, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        #layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        return nn.Sequential(*layers), ndim


    def intermediate_block(self, resl):

        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2, resl - 1)), int(pow(2, resl - 1)), int(pow(2, resl)),
                                                       int(pow(2, resl)))
        ndim = self.ngf / 2**(resl-2)

        ndim = int(ndim)

        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode=self.upsam_mode))  # scale up by factor of 2.0

        layers = deconv(layers, ndim*2, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)
        #layers = deconv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, self.flag_pixelwise)

        return nn.Sequential(*layers), ndim, layer_name


    def to_rgb_block(self, c_in):
        layers = []
        layers = deconv(layers, c_in, self.nc, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn,
                        self.flag_pixelwise, only=True)
        if self.flag_tanh:  layers.append(nn.Tanh())
        return nn.Sequential(*layers)


    def grow_network(self, resl):
        # we make new network since pytorch does not support remove_module()
        new_model = nn.Sequential()
        for name, module in self.model.named_children():
            if not name == 'to_rgb_block':
                new_model.add_module(name, module)  # make new structure and,
                new_model[-1].load_state_dict(module.state_dict())  # copy pretrained weights

        if resl >= 3 and resl <= 9:
            print('growing network[{}x{} to {}x{}]. It may take few seconds...'.format(int(pow(2, resl - 1)),
                                                                                       int(pow(2, resl - 1)),
                                                                                       int(pow(2, resl)),
                                                                                       int(pow(2, resl))))
            low_resl_to_rgb = deepcopy_module(self.model, 'to_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_upsample', nn.Upsample(scale_factor=2, mode=self.upsam_mode))
            prev_block.add_module('low_resl_to_rgb', low_resl_to_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_block', inter_block)
            next_block.add_module('high_resl_to_rgb', self.to_rgb_block(ndim))

            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)

    def flush_network(self):
        try:
            print('flushing network... It may take few seconds...')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_to_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_to_rgb')

            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name != 'concat_block' and name != 'fadein_block':
                    new_model.add_module(name, module)  # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())  # copy pretrained weights

            # now, add the high resolution block.
            new_model.add_module(self.layer_name, high_resl_block)
            new_model.add_module('to_rgb_block', high_resl_to_rgb)
            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1, 1, 1, 1))
        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_sigmoid = config.flag_sigmoid
        self.flag_gdrop = config.flag_gdrop
        self.nz = config.nz
        self.nc = config.nc
        self.ndf = config.ndf
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_dis()

    def last_block(self):
        # add minibatch_std_concat_layer later.
        ndim = self.ndf
        layers = []
        # layers.append(minibatch_std_concat_layer())
        #layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        layers = conv(layers, ndim, ndim, 4, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False, gdrop=self.flag_gdrop)
        layers = linear(layers, ndim, 1, sig=self.flag_sigmoid, wn=self.flag_wn)
        return nn.Sequential(*layers), ndim

    def intermediate_block(self, resl):

        layer_name = 'intermediate_{}x{}_{}x{}'.format(int(pow(2, resl)), int(pow(2, resl)), int(pow(2, resl - 1)),
                                                       int(pow(2, resl - 1)))
        ndim = self.ndf / 2**(resl-2)
        ndim = int(ndim)
        layers = []

        #layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False)
        layers = conv(layers, ndim, ndim * 2, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False, gdrop=self.flag_gdrop)

        layers.append(nn.AvgPool3d(kernel_size=2))  # scale up by factor of 2.0
        return nn.Sequential(*layers), ndim, layer_name

    def from_rgb_block(self, ndim):
        layers = []
        layers = conv(layers, self.nc, ndim, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_wn, pixel=False, gdrop=self.flag_gdrop)
        return nn.Sequential(*layers)

    def get_init_dis(self):
        model = nn.Sequential()
        last_block, ndim = self.last_block()
        model.add_module('from_rgb_block', self.from_rgb_block(ndim))
        model.add_module('last_block', last_block)
        self.module_names = get_module_names(model)
        return model

    def grow_network(self, resl):

        if resl >= 3 and resl <= 9:
            print('growing network[{}x{} to {}x{}]. It may take few seconds...'.format(int(pow(2, resl - 1)),
                                                                                       int(pow(2, resl - 1)),
                                                                                       int(pow(2, resl)),
                                                                                       int(pow(2, resl))))
            low_resl_from_rgb = deepcopy_module(self.model, 'from_rgb_block')
            prev_block = nn.Sequential()
            prev_block.add_module('low_resl_downsample', nn.AvgPool3d(kernel_size=2))
            prev_block.add_module('low_resl_from_rgb', low_resl_from_rgb)

            inter_block, ndim, self.layer_name = self.intermediate_block(resl)
            next_block = nn.Sequential()
            next_block.add_module('high_resl_from_rgb', self.from_rgb_block(ndim))
            next_block.add_module('high_resl_block', inter_block)

            new_model = nn.Sequential()
            new_model.add_module('concat_block', ConcatTable(prev_block, next_block))
            new_model.add_module('fadein_block', fadein_layer(self.config))

            # we make new network since pytorch does not support remove_module()
            names = get_module_names(self.model)
            for name, module in self.model.named_children():
                if not name == 'from_rgb_block':
                    new_model.add_module(name, module)  # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())  # copy pretrained weights
            self.model = None
            self.model = new_model
            self.module_names = get_module_names(self.model)

    def flush_network(self):
        try:
            print('flushing network... It may take few seconds...')
            # make deep copy and paste.
            high_resl_block = deepcopy_module(self.model.concat_block.layer2, 'high_resl_block')
            high_resl_from_rgb = deepcopy_module(self.model.concat_block.layer2, 'high_resl_from_rgb')

            # add the high resolution block.
            new_model = nn.Sequential()
            new_model.add_module('from_rgb_block', high_resl_from_rgb)
            new_model.add_module(self.layer_name, high_resl_block)

            # add rest.
            for name, module in self.model.named_children():
                if name != 'concat_block' and name != 'fadein_block':
                    new_model.add_module(name, module)  # make new structure and,
                    new_model[-1].load_state_dict(module.state_dict())  # copy pretrained weights

            self.model = new_model
            self.module_names = get_module_names(self.model)
        except:
            self.model = self.model

    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x


# defined for code simplicity.
def Trans_conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, only=False):
    if wn:
        layers.append(equalized_deconv3d(c_in, c_out, k_size, stride, pad))
    else:
        layers.append(nn.ConvTranspose3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm3d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers


# defined for code simplicity.
def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, only=False):
    if wn:
        layers.append(equalized_conv3d(c_in, c_out, k_size, stride, pad))
    else:
        layers.append(nn.Conv3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm3d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers


def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, wn=False, pixel=False, gdrop=True, only=False):
    if gdrop:       layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:
        layers.append(equalized_conv3d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:
        layers.append(nn.Conv3d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm3d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers


def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:
        layers.append(equalized_linear(c_in, c_out))
    else:
        layers.append(Linear(c_in, c_out))
    if sig:     layers.append(nn.Sigmoid())
    return layers


def deepcopy_module(module, target):
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)  # make new structure and,
            new_module[-1].load_state_dict(m.state_dict())  # copy weights
    return new_module


def soft_copy_param(target_link, source_link, tau):
    ''' soft-copy parameters of a link to another link. '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0 - tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))


def get_module_names(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


if __name__ == "__main__":

    dis = Discriminator(config)
    dis.grow_network(3)
    dis.flush_network()
    dis.grow_network(4)
    dis.flush_network()
    dis.grow_network(5)
    dis.flush_network()
    dis.grow_network(6)
    dis.flush_network()
    dis.grow_network(7)
    # dis.flush_network()
    dis = dis.cuda()
    print(dis)
    for _ in range(100000):
        input = torch.randn(1, 1, 128, 128, 128).cuda()
        ouput = dis(input)
        print(ouput.shape)

    # input = torch.rand(16, 1, 4, 4, 4)
    # print(dis(input).shape)
    #
    # dis.grow_network(3)
    # dis.flush_network()
    # print(dis)

    # print(config)
    # generator = Generator(config)
    # generator.grow_network(3)
    # generator.flush_network()
    # generator.grow_network(4)
    # generator.flush_network()
    # generator.grow_network(5)
    # generator.flush_network()
    # generator.grow_network(6)
    # generator.flush_network()
    # generator.grow_network(7)
    # generator.flush_network()
    # generator.cuda()
    #
    # for _ in range(100000):
    #     input = torch.randn(1, 100).cuda()
    #     ouput = generator(input)
    #     print(ouput.shape)

    # input = torch.randn(10, 512)
    # output = generator(input)
    # print(output.shape)
    # print(generator)
    #
    # generator.grow_network(3)
    # input = torch.randn(10, 512)
    # output = generator(input)
    # print(output.shape)
    # print(generator)
    #
    #
    # generator.flush_network()
    # input = torch.randn(10, 512)
    # output = generator(input)
    # print(output.shape)
    # print(generator)
    #
    # generator.grow_network(4)
    # input = torch.randn(10, 512)
    # output = generator(input)
    # print(output.shape)
    # print(generator)
    #
    # generator.flush_network()
    # input = torch.randn(10, 512)
    # output = generator(input)
    # print(output.shape)
    # print(generator)