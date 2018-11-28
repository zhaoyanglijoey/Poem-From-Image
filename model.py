import torch
import torchvision.models as models
import torch.nn as nn
import os 

class VGG16_fc7_object(nn.Module):
    def __init__(self):
        super(VGG16_fc7_object, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.require_grad = False
        self.fc7 = nn.Sequential(list(self.vgg.children())[0], list(self.vgg.children())[1][0])

    def forward(self, x):
        return self.fc7(x)

class PlacesCNN(nn.Module):
    def __init__(self, arch='resnet50'):
        super(PlacesCNN, self).__init__()

        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        layers = list(model.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.backbone(x)