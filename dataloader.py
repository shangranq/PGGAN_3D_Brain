from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from utils import read_txt, read_csv
import random
import copy
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image


class Data(Dataset):
    def __init__(self, Data_dir, seed=1000):
        random.seed(seed)
        self.Data_dir = Data_dir
        self.Data_list, self.Label_list = read_csv('ADNI.csv')

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + '.npy').astype(np.float32)
        data = np.expand_dims(data, axis=0)
        return data, label


def save_image(array, filename, nrow, ncol):
    count, w, h = array.shape
    whole_image = np.zeros((nrow*w, ncol*h))
    for i in range(nrow):
        for j in range(ncol):
            index = i * nrow + j
            whole_image[i*w:(i+1)*w, j*h:(j+1)*h] = array[index]
    plt.imshow(whole_image, vmin=-1, vmax=1)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    array = []
    dataloader = DataLoader(Data('./mri/resl4/'), batch_size=1, shuffle=True)
    for data, label in dataloader:
        print(data.shape, label)
        array.append(data.squeeze().data.numpy())
        if len(array) == 10:
            break
    array = np.array(array)
    save_image(array, './mri/real4.png', 2, 5)
