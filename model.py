import torch
import torchvision.models as models
import torch.nn as nn
import os
from pytorch_pretrained_bert import BertModel

class PoemImageEmbedModel(nn.Module):
    def __init__(self, device, alpha=0.2):
        super(PoemImageEmbedModel, self).__init__()
        self.img_embedder = ImageEmbed()
        self.poem_embedder = PoemEmbed()
        self.alpha = alpha
        self.device = device

    def normalize(self, t):
        out = t / torch.norm(t, dim=1, keepdim=True)
        return out

    def forward(self, img1, ids1, mask1, img2, ids2, mask2):
        img_embed1 = self.img_embedder(img1)
        poem_embed1 = self.poem_embedder(ids1, mask1)
        img_embed2 = self.img_embedder(img2)
        poem_embed2 = self.poem_embedder(ids2, mask2)

        return self.rank_loss(img_embed1, poem_embed1, img_embed2, poem_embed2)

    def rank_loss(self, img_embed1, poem_embed1, img_embed2, poem_embed2):
        img_embed1 = self.normalize(img_embed1)
        poem_embed1 = self.normalize(poem_embed1)
        img_embed2 = self.normalize(img_embed2)
        poem_embed2 = self.normalize(poem_embed2)

        zero_tensor = torch.zeros(img_embed1.size(0)).to(self.device)
        loss1 = torch.max(self.alpha - torch.sum(img_embed1 * poem_embed1, dim=1) + \
               torch.sum(img_embed1 * poem_embed2, dim=1), zero_tensor)
        loss2 = torch.max(self.alpha - torch.sum(poem_embed2 * img_embed2, dim=1) + \
               torch.sum(poem_embed2 * img_embed1, dim=1), zero_tensor)
        loss = torch.mean(loss1 + loss2)

        return loss


class PoemEmbed(nn.Module):
    def __init__(self, embed_dim=512):
        super(PoemEmbed, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_dim)

    def forward(self, ids, mask):
        _, bert_feature = self.bert(ids, attention_mask=mask)
        return self.linear(bert_feature)

class ImageEmbed(nn.Module):
    def __init__(self, embed_dim =512):
        super(ImageEmbed, self).__init__()
        self.object_feature = Res50_object()
        self.scene_feature = PlacesCNN()
        # TODO sentiment feature
        self.linear = nn.Linear(2048*2, embed_dim)

    def forward(self, x):
        features = torch.cat([self.object_feature(x), self.scene_feature(x)], dim=1)
        return self.linear(features)

class VGG16_fc7_object(nn.Module):
    def __init__(self):
        super(VGG16_fc7_object, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.fc7 = nn.Sequential(list(self.vgg.children())[0], list(self.vgg.children())[1][0])

    def forward(self, x):
        return self.fc7(x)

class Res50_sentiment(nn.Module):
    def __init__(self):
        super(Res50_sentiment, self).__init__()
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 3)
        # self.fc2 = nn.Linear(512, 3)
        # self.relu = nn.ReLU()

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        # out = self.fc2(self.relu(self.fc1(out)))
        out = self.fc1(out)
        return out

class Res50_object(nn.Module):
    def __init__(self):
        super(Res50_object, self).__init__()
        ResNet50 = models.resnet50(pretrained=True)
        for param in ResNet50.parameters():
            param.requires_grad = False
        modules = list(ResNet50.children())[:-1]
        self.feature_layer = nn.Sequential(*modules)

    def forward(self, x):
        # [ , 2048]
        out = self.feature_layer(x)
        return out.view(out.size(0), -1)

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

        for param in model.parameters():
            param.requires_grad = False

        layers = list(model.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        
        
    def forward(self, x):
        return self.backbone(x).view(x.size(0), -1)