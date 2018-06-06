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
parser.add_argument('--gpu_ids', default='1', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='200', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()
name = opt.name
epoch_num = 200

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

'''
the datasets have been preprocessed by opencv after some basic data analysis and we adjusted some
features of some certain pictures and they were added in the train datas folders which are marked by "Newxxx.jpg"
'''
transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((234,224), interpolation=3),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop((224, 224)),
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
train_loader = DataLoader(dataset=train_data, batch_size=28, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

use_gpu = torch.cuda.is_available()
#-----------------create the Net and training------------------------


#resnet152 = models.resnet152(pretrained = True)
model = senet.se_resnet152(1000, pretrained = None)


'''
the codes of se_resnet are from https://github.com/hujie-frank/SENet who originally built the network framework
'''
#___load the previous model for the first train___#
Senet = senet.se_resnet152(1000, pretrained = 'imagenet')
pretrained_dict = Senet.state_dict()
model_dict = model.state_dict()
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.last_linear = nn.Linear(2048,100)
#___load the pretrained model___#
#model = tools.load_network(model, name, 100)

print(model)

if use_gpu:
    model = model.cuda()

# optimizer = torch.optim.Adam(model.parameters())
ignored_params = list(map(id, model.last_linear.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.001},
        {'params': model.last_linear.parameters(), 'lr': 0.001}
        # {'params': model.classifier.parameters(), 'lr': 0.1}
    ], weight_decay=5e-4, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()

TrainOrTest = 'train'
for epoch in range(epoch_num):
    print('\n\nepoch {}/{}'.format(epoch + 1, epoch_num))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:

        if use_gpu:
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())
        else:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        print('Batch Loss: {:.6f}'.format(loss.data[0]))
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch Loss: {:.6f}, Acc: {:.6f}'.format(train_loss , train_acc / (len(train_data))))
    if epoch % 20 == 19:
	#save that network automatically
        tools.save_network(model, epoch+1, name, gpu_ids[0])
    #modify learning rate
    if epoch % 20 == 19:
        tools.adjust_learning_rate(optimizer)






