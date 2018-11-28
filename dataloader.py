import torch
from torch.utils.data import Dataset
from PIL import Image
import os

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
        img = Image.open(os.path.join(self.img_dir, '{}.jpg'.format(d['id'])))
        img = self.transform(img)

        return img, word_ind
