import torch
import torch.nn as nn
from torchvision import transforms, models


class AlexNet(torch.nn.Module):
    def __init__(self, classnum):

        super(AlexNet, self).__init__()
        model_AlexNet = models.alexnet(pretrained=True, num_classes=classnum)
        self.model = model_AlexNet

    def forward(self, x):
        output = self.model(x)
        return output

class VGGNet(torch.nn.Module):
    def __init__(self, classnum):

        super(VGGNet, self).__init__()
        model_VGGNet = models.vgg16(pretrained=True, num_classes=classnum)
        self.model = model_VGGNet

    def forward(self, x):
        output = self.model(x)
        return output


class GoogLenet(torch.nn.Module):
    def __init__(self, classnum):
        super(GoogLenet, self).__init__()
        model_GoogLenet = models.inception_v3(pretrained=True, num_classes=classnum)
        self.model = model_GoogLenet

    def forward(self, x):
        output = self.model(x)
        return output



class Res50Net(torch.nn.Module):
    def __init__(self, classnum):

        super(Res50Net, self).__init__()
        model_Res50 = models.resnet50(pretrained=True)
        model_Res50.fc = nn.Linear(2048, classnum)
        self.model = model_Res50

    def forward(self, x):
        output = self.model(x)
        return output
