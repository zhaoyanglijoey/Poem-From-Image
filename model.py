import torch
import torchvision.models as models
import torch.nn as nn
import os
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertForMaskedLM
from dataloader import aligned_ids, convert_to_bert_ids, convert_to_bert_ids_no_sep
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions import Categorical
from copy import deepcopy


def normalize(t):
    out = t / torch.norm(t, dim=-1, keepdim=True)
    return out

class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_labels = 2, feature_size=512):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size//2, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size+feature_size, num_labels)
        self.dropout = nn.Dropout(0.2)


    def forward(self, seq, lengths, features):
        features = normalize(features)

        embeddings = self.embed(seq)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        _, (hidden, _) = self.rnn(packed)
        hidden = hidden.transpose(0, 1).contiguous().view(-1, self.hidden_size)
        output = torch.cat([hidden, features], dim=-1)
        output = self.dropout(output)
        output = self.classifier(output)
        return output


class DecoderRNN(nn.Module):
    """
    Example training file: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/train.py
    Example sampling file: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py
    """
    def __init__(self, embed_size, hidden_size, vocab_size, device, max_seq_length=70,
                 sos_index=1, eos_index=2, feature_size = 512):
        """
        Set the hyper-parameters and build the layers."
        :param embed_size: word embedding size
        :param hidden_size: hidden size of GRU. Make sure equal to size of image features
        :param vocab_size: size of vocabulary
        :param max_seq_length: the number of words at most
        :param sos_index: start of sentence: index (usually 1)
        """
        super(DecoderRNN, self).__init__()
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(0.2)
        self.linear.weight = self.embed.weight # tie weights
        self.rnn_cell = nn.LSTMCell(feature_size, hidden_size)

    def forward(self, features, poem_word_indices, lengths):
        """
        Decode image feature vectors and generates captions.
        :param features: image features. (batch_size, feature_size)
        :param poem_word_indices: indices of words in captions including <SOS> and <EOS>. (batch_size, max_length)
        :param lengths: lengths of captions including <SOS> and <EOS> (batch_size, )
        :return: Distribution. (words_in_batch, size_vocab)
        """
        """Decode image feature vectors and generates captions."""
        features = normalize(features)
        features = self.dropout(features)
        (h, c) = self.rnn_cell(features)
        embeddings = self.embed(poem_word_indices)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # make sure image features size equal to GRU hidden_size
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        rnn_outputs, (_, _) = self.rnn(packed, (h, c))
        (rnn_outputs_unpack, unpack_lengths) = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=poem_word_indices.size(1)-1)
        rnn_outputs_unpack = self.dropout(rnn_outputs_unpack)
        # rnn_outputs_repack = pack_padded_sequence(rnn_outputs_unpack, lengths, batch_first=True)
        outputs = self.linear(rnn_outputs_unpack)
        return outputs

    def sample_beamsearch(self, features, beamsize=10, k=3, temperature=1):
        features = normalize(features)
        prev_id = torch.tensor(self.sos_index, dtype=torch.long).to(self.device)
        with torch.no_grad():
            (h, c) = self.rnn_cell(features)
            h = h.unsqueeze(0)
            c = c.unsqueeze(0)
            beams = [(0, h, c, [prev_id])]
            for i in range(self.max_seq_length):
                tmp = []
                for avg_log_prob, h, c, history in beams:
                    if history[-1].item() == self.eos_index:
                        tmp.append((avg_log_prob, h, c, history))
                        continue
                    prev_id = history[-1].unsqueeze(0).unsqueeze(0)
                    inputs = self.embed(prev_id)
                    lstm_outputs, (h, c) = self.rnn(inputs, (h, c))  # lstm_outputs: (batch_size, 1, hidden_size)
                    outputs = self.linear(lstm_outputs.squeeze(1))
                    outputs = outputs[0]
                    weights = F.softmax(outputs / temperature, dim=-1)
                    preds = torch.multinomial(weights, k)
                    # preds = m.sample(torch.Size(k))
                    for pred in preds:
                        new_history = deepcopy(history) + [pred]
                        log_prob = weights[pred]
                        avg_log_prob = (avg_log_prob * len(history) + log_prob) / len(new_history)
                        tmp.append((avg_log_prob, h, c, new_history))
                tmp.sort(reverse=True, key=lambda t:t[0].item())
                beams = tmp[:beamsize]
        return beams[0][3]


    def sample(self, features, temperature = 1):
        """
        Generate captions for given image features using greedy search.
        :param features: image features. (batch_size, feature_size)
        :return: contents of poem. (batch_size, max_seq_length)
        """
        features = normalize(features)

        batch_size = features.shape[0]
        # sampled_ids = [torch.full((batch_size, ), 56, dtype=torch.long).to('cuda')]
        sampled_ids = []
        # use <sos> as init input
        start = torch.full((batch_size, 1), self.sos_index, dtype=torch.int).long().to(self.device)  # start symbol index is 1
        inputs = self.embed(start)  # inputs: (batch_size, 1, embed_size)

        # use img features as init hidden_states
        # hidden_states = (features.unsqueeze(0), features.unsqueeze(0))  # add one dimension as num_layers * num_directions (which is 1)
        (h, c) = self.rnn_cell(features)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        for i in range(self.max_seq_length):
            lstm_outputs, (h, c) = self.rnn(inputs, (h, c))  # lstm_outputs: (batch_size, 1, hidden_size)
            outputs = self.linear(lstm_outputs.squeeze(1))  # outputs:  (batch_size, vocab_size)
            # _, predicted = outputs.max(1)  # predicted: (batch_size)
            weights = F.softmax(outputs/temperature, dim=1)
            predicted = torch.multinomial(weights, 1).squeeze(-1)
            # predicted = torch.sort(outputs, dim=1, descending=True)[1][:, 1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class BertLMGenerator(nn.Module):
    def __init__(self, vocab_size):
        super(BertLMGenerator, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, ids, attn_mask, masked_lm_labels=None):

        return self.bert(ids, attention_mask=attn_mask, masked_lm_labels=masked_lm_labels)


    def generate(self, feature, max_gen_len, basic_tokenizer, tokenizer,
                 word2idx, idx2word, max_seq_len, device, temperature=1):
        seq = ''
        feature = feature.unsqueeze(0).to(device)
        pred_words = []
        for i in range(max_gen_len):
            id, attn_mask, length = convert_to_bert_ids_no_sep(seq, tokenizer, max_seq_len)
            id = id.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            outputs = self.forward(id, attn_mask)
            # print(attn_mask[0].sum()-1)
            # pred_ind = torch.argmax(outputs[0][length-1], dim=-1).long().cpu().item()
            weights = F.softmax(outputs[:, length-1]/temperature, dim=1)
            pred_ind = torch.multinomial(weights, 1).squeeze(-1).cpu().item()
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
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.linear = nn.Linear(768+512, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, ids, mask, align_mask, feature):
        encoded_layers, _ = self.bert(ids, attention_mask=mask, output_all_encoded_layers=False)
        align_encoded = encoded_layers[align_mask == 1]
        align_encoded = self.dropout(align_encoded)
        feature = normalize(feature)
        feature_exp = feature.unsqueeze(1).repeat(1, encoded_layers.size(1), 1)
        feature_exp = feature_exp[align_mask == 1]
        outputs = self.linear(torch.cat([align_encoded, feature_exp], dim=-1))
        return outputs

    def generate(self, feature, max_gen_len, basic_tokenizer, tokenizer,
                 word2idx, idx2word, max_seq_len, device, temperature = 1):
        seq = ''
        pred_words = []
        for i in range(max_gen_len):
            id, attn_mask, align_mask, word_ind, length_m1 = aligned_ids(
            seq, basic_tokenizer, tokenizer, word2idx, max_seq_len)
            id = id.unsqueeze(0).to(device)
            attn_mask = attn_mask.unsqueeze(0).to(device)
            align_mask = align_mask.unsqueeze(0).to(device)
            outputs = self.forward(id, attn_mask, align_mask, feature)
            # pred_ind = torch.argmax(outputs, dim=-1)
            # pred_words = [idx2word[idx.item()] for idx in pred_ind]
            # pred_ind = torch.argmax(outputs[-1], dim=-1)
            weights = torch.nn.functional.softmax(outputs[-1] / temperature, dim=-1)
            pred_ind = torch.multinomial(weights, 1).squeeze(-1)
            pred_words.append(idx2word[pred_ind.item()])
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

# class VGG16_fc7_object(nn.Module):
#     def __init__(self):
#         super(VGG16_fc7_object, self).__init__()
#         self.vgg = models.vgg16(pretrained=True)
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#         self.fc7 = nn.Sequential(list(self.vgg.children())[0], list(self.vgg.children())[1][0])
#
#     def forward(self, x):
#         return self.fc7(x)

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