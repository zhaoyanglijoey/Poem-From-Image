import torch
from torch.utils.data import Dataset
from PIL import Image
import os, random, sys

def convert_to_bert_ids(seq, tokenizer, max_seq_len):
    tokens = tokenizer.tokenize(seq)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[0:(max_seq_len-2)]

    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padded_ids = [0] * max_seq_len
    padded_ids[:len(ids)] = ids
    mask = [0] * max_seq_len
    mask[:len(ids)] = [1] * len(ids)

    # assert len(padded_ids) == max_seq_len
    # assert len(mask) == max_seq_len

    padded_ids = torch.tensor(padded_ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)

    return padded_ids, mask

class PoemImageDataset(Dataset):
    def __init__(self, data, img_dir, word2idx, transform = None, train=True):
        num_train = int(len(data) * 0.95)
        self.img_dir = img_dir
        self.transform = transform
        self.word2idx = word2idx
        if train:
            self.data = data[:num_train]
        else:
            self.data = data[num_train:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            img: [ , 224, 224, 3] tensor
            word_ind: [ , T] word indices tensor
        '''

        d = self.data[index]
        poem = d['poem'].replace('\n', ' \n ').split(' ')
        word_ind = [self.word2idx[word] for word in poem]
        word_ind = torch.tensor(word_ind, dtype=torch.int64)
        img = Image.open(os.path.join(self.img_dir, '{}.jpg'.format(d['id']))).convert('RGB')
        img = self.transform(img)

        return img, word_ind

class PoemImageEmbedDataset(Dataset):
    def __init__(self, data, img_dir, tokenizer, max_seq_len, transform = None):
        super(PoemImageEmbedDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample1 = self.data[index]
        sample2_idx = random.randrange(len(self.data))
        while sample2_idx == index:
            sample2_idx = random.randrange(len(self.data))
        sample2 = self.data[sample2_idx]

        img1 = Image.open(os.path.join(self.img_dir, '{}.jpg'.format(sample1['id']))).convert('RGB')
        img1 = self.transform(img1)
        img2 = Image.open(os.path.join(self.img_dir, '{}.jpg'.format(sample2['id']))).convert('RGB')
        img2 = self.transform(img2)

        ids1, mask1 = convert_to_bert_ids(sample1['poem'], self.tokenizer, self.max_seq_len)
        ids2, mask2 = convert_to_bert_ids(sample2['poem'], self.tokenizer, self.max_seq_len)


        return img1, ids1, mask1, img2, ids2, mask2


class VisualSentimentDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        super(VisualSentimentDataset, self).__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        id = entry['id']
        img = Image.open(os.path.join(self.img_dir, '{}.jpg'.format(id))).convert('RGB')
        img = self.transform(img)

        # level = {'Highly negative': 0, 'Negative': 0,0 'Neutral': 1, 'Positive': 2, 'Highly positive': 2}
        # label = level[entry['label']]
        # label = torch.tensor(label, dtype=torch.long)

        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        positive_emotions = ['amusement', 'awe', 'excitement', 'contentment']
        negative_emotions = ['anger', 'disgust', 'fear', 'sadness']

        if entry['disagrees'] > entry['agrees']:
                label = label_map['neutral']
        else:
            if entry['emotion'] in positive_emotions:
                label = label_map['positive']
            elif entry['emotion'] in negative_emotions:
                label = label_map['negative']
            else:
                print('Error: unknown emotion {}'.format(entry['emotion']))
                exit(-1)
        label = torch.tensor(label, dtype=torch.long)

        return img, label
