import argparse
import os
import zipfile
import glob
import time
import pickle
import pprint
import math


import matplotlib.pyplot as plt
import PIL

import numpy as np
import pandas as pd
import sklearn.model_selection

import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as f

import torchvision
import torch.utils.data

# for process visualization
import tqdm
# https://stackoverflow.com/a/59345557/12224183
# https://calmcode.io/tqdm/nested-loops.html

# # for TPU
# assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
# import torch_xla
# import torch_xla.core.xla_model as xm


class DataSet(torch.utils.data.Dataset):

    def __init__(self, cache, transform=None):
        self.cache = cache
        self.transform = transform
        self.cache_length = len(self.cache)

    # dataset length
    def __len__(self):
        return self.cache_length

    # load an one of images
    def __getitem__(self, idx):
        img_transformed = self.transform(self.cache[idx][0])
        label = self.cache[idx][1]
        return img_transformed, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 512, 2048)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(2048, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def factorize(n):
    temp = math.floor(math.sqrt(n))
    while n % temp != 0:
        temp -= 1
    return temp, int(n / temp)


def visualize(name):
    fig = plt.figure()
    fig.suptitle(str(name).title() + ' Test')
    model.eval()
    path = '/content/drive/MyDrive/kaggle-dogs-vs-cats-pytorch/data/' + name + '/'
    mytest_list = [path + _ for _ in os.listdir(path)]
    mytest_num = len(mytest_list)
    height, width = factorize(mytest_num)
    mytest_cache = [(PIL.Image.open(_), _.split('/')[-1].split('.')[0]) for _ in mytest_list]
    mytest_data = DataSet(mytest_cache, transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.Resize((224, 224)),
                                                  torchvision.transforms.ToTensor()
    ]))

    with torch.no_grad():
        for i, _ in enumerate(mytest_data):
            data, _ = torch.utils.data.DataLoader(dataset=_)
            if data.shape[1] == 4:  # '.png' images have 4 channels
                data = data[:, :3, :, :]
            data = data.to(device)  # shape: torch.Size([1, 3, 224, 224])
            pred = model(data)  # tensor([[-1.2971, -0.2252]], device='cuda:0')
            pred = f.softmax(pred, dim=1)[:, 1][0]
            ax = fig.add_subplot(height, width, i+1)
            ax.set_axis_off()  # https://stackoverflow.com/a/52776192/12224183
            plt.imshow(mytest_cache[i][0])
            ax.set_title('{}% dog'.format(int(pred*100)) if pred > 0.5 else '{}% cat'.format(int(100*(1-pred))))


model = torch.load('assets/20210604081904.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Simple argparse example wanted: 1 argument, 3 results
# https://stackoverflow.com/questions/7427101/simple-argparse-example-wanted-1-argument-3-results
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', help='Description for foo argument', required=True)
args = vars(parser.parse_args())
visualize(args['dir'])
