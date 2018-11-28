import torch
import torchvision.models as models
import torch.nn as nn

class VGG16_fc7_object(nn.Module):
    def __init__(self):
        super(VGG16_fc7_object, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.require_grad = False
        self.fc7 = nn.Sequential(list(self.vgg.children())[0], list(self.vgg.children())[1][0])

    def forward(self, x):
        return self.fc7(x)