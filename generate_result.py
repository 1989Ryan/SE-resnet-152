# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.optim as optim
import tools
import model
import senet
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
root="./datasets/"

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='seresnet', type=str, help='output model name')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='20', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()
name = opt.name

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
print(gpu_ids)



# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, TrainOrTest = 'train', target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if TrainOrTest == 'train':
                imgs.append((words[0],int(words[1])-1)) #start from 0 
            else:
                imgs.append(words[0])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        if TrainOrTest == 'train':
            fn, label = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            return img,label
        else:
            fn = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.imgs)


transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((224,224), interpolation=3),
        transforms.RandomRotation(degrees=5),
        #transforms.RandomCrop((224, 224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform_test = transforms.Compose(
    [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((224,224), interpolation=3),
        #transforms.RandomRotation(degrees=15),
        #transforms.CenterCrop((224, 224)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_data=MyDataset(txt=root+'newtrain.txt', transform=transform, TrainOrTest='train')
test_data=MyDataset(txt=root+'newtest.txt', transform=transform_test, TrainOrTest='test')
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)
use_gpu = torch.cuda.is_available()

#-------------------------generate the result------------------------------#
TrainOrTest = 'test'
test_model = senet.se_resnet152(100, pretrained = None)
#test_model.last_linear = nn.Linear(2048,100)
test_model = tools.load_senetwork(test_model, name, opt.which_epoch)
model = test_model.eval()
if use_gpu:
    model = model.cuda()

result = {}
k=0
for batch_x in test_loader:
    if use_gpu:
        batch_x = Variable(batch_x.cuda())
    else:
        batch_x = Variable(batch_x)
    out = model(batch_x)
    result[test_data.imgs[k].split('/')[-1]] = torch.max(out, dim=1)[1].data[0]
    k += 1

#--------------------------------write the text------------------------------------#
with open('result.txt', 'w') as retxt:
    for k in range(len(result)):
        retxt.write(test_data.imgs[k].split('/')[-1]+ ' ' + str(result[test_data.imgs[k].split('/')[-1]]+1)+'\n')


