import argparse
import os
import torch
import torch.optim as optim
import tools
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def save_network(network, epoch_label, name, gpu_ids):
    save_filename = 'net_%s.pth' % epoch_label
    file_path = os.path.join('./models', name)
    save_path = os.path.join('./models', name, save_filename)
    print(save_path)
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    torch.save(network.cpu().state_dict(), save_path)
    print('saved')
    if torch.cuda.is_available:
        network.cuda(gpu_ids)


def load_network(network, name, which_epoch):
    save_path = os.path.join('./models', name, 'net_%s.pth' % which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

def load_senetwork(network, name, which_epoch):
    save_path = os.path.join('./models', name, 'net_%s.pth' % which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def adjust_learning_rate(optimizer, decay_rate = 0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

