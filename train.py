import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer
import os, sys, time
from dataloader import PoemImageDataset, PoemImageEmbedDataset
from model import VGG16_fc7_object, PoemImageEmbedModel
import json
from util import load_vocab_json, build_vocab
from tqdm import tqdm


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('data/multim_poem.json') as f, open('data/unim_poem.json') as unif:
        multim = json.load(f)
        unim = json.load(unif)

    word2idx, idx2word = build_vocab(unim)
    num_train = int(len(multim) * 0.95)
    train_data = multim[:num_train]
    test_data = multim[num_train:]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    img_dir = 'data/image'
    train_set = PoemImageDataset(multim, img_dir, word2idx, transform=train_transform, train=True)
    test_set = PoemImageDataset(multim, img_dir, word2idx, transform=test_transform, train=True)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1)

    model = VGG16_fc7_object() # TODO
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 3e-4)

    sys.stderr.write('training...\n')
    for i, (poem, img) in enumerate(tqdm(train_loader)):
        # print(poem)
        # print(img)
        pass


if __name__ == '__main__':
    main()
