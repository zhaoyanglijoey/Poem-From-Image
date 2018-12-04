import torch
import torchvision.models as models
import torch.nn as nn
import os
from pytorch_pretrained_bert import BertModel, BertForMaskedLM
from dataloader import aligned_ids, convert_to_bert_ids, convert_to_bert_ids_no_sep

class BertLMGenerator(nn.Module):
    def __init__(self, vocab_size):
        super(BertLMGenerator, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, ids, attn_mask, masked_lm_labels=None):

        return self.bert(ids, attention_mask=attn_mask, masked_lm_labels=masked_lm_labels)


    def generate(self, feature, max_gen_len, basic_tokenizer, tokenizer, word2idx, idx2word, max_seq_len, device):
        seq = ''
        feature = feature.unsqueeze(0).to(device)
        pred_words = []
        for i in range(max_gen_len):
            id, attn_mask = convert_to_bert_ids_no_sep(seq, tokenizer, max_seq_len)
            id = id.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            outputs = self.forward(id, attn_mask)
            # print(attn_mask[0].sum()-1)
            pred_ind = torch.argmax(outputs[0][attn_mask[0].sum()-2], dim=-1).long().cpu().item()
            next_word = tokenizer.convert_ids_to_tokens([pred_ind])[0]
            if next_word == '[SEP]':
                return pred_words
            pred_words.append(next_word)
            seq = ' '.join(pred_words)

        return pred_words

class BertGenerator(nn.Module):
    def __init__(self, vocab_size):
        super(BertGenerator, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768+512, vocab_size)
        self.dropout = nn.Dropout()

    def forward(self, ids, mask, align_mask, feature):
        encoded_layers, _ = self.bert(ids, attention_mask=mask, output_all_encoded_layers=False)
        align_encoded = encoded_layers[align_mask == 1]
        align_encoded = self.dropout(align_encoded)
        feature_exp = feature.unsqueeze(1).repeat(1, encoded_layers.size(1), 1)
        feature_exp = feature_exp[align_mask == 1]
        outputs = self.linear(torch.cat([align_encoded, feature_exp], dim=-1))
        return outputs

    def generate(self, feature, max_gen_len, basic_tokenizer, tokenizer, word2idx, idx2word, max_seq_len, device):
        seq = ''
        feature = feature.unsqueeze(0).to(device)
        for i in range(max_gen_len):
            id, attn_mask, align_mask, word_ind, length_m1 = aligned_ids(
            seq, basic_tokenizer, tokenizer, word2idx, max_seq_len)
            id = id.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            align_mask = align_mask.unsqueeze(0).to(device)
            outputs = self.forward(id, attn_mask, align_mask, feature)
            pred_ind = torch.argmax(outputs, dim=-1)
            pred_words = [idx2word[idx.item()] for idx in pred_ind]
            if pred_words[-1] == '[SEP]':
                return pred_words
            seq = ' '.join(pred_words)

        return pred_words



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
        self.sentiment_feature = Res50_sentiment()
        self.linear = nn.Linear(2048*3, embed_dim)

    def forward(self, x):
        features = torch.cat([self.object_feature(x),
                              self.scene_feature(x),
                              self.sentiment_feature.get_feature(x)], dim=1)
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

    def get_feature(self, x):
        return self.backbone(x).view(x.size(0), -1)

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