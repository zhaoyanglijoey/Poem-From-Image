import torch
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer, BertModel
import os, sys, time
import dataloader
from model import BertGenerator, BertLMGenerator, PoemImageEmbedModel
import json, pickle, glob2
from tqdm import tqdm
import argparse
import util
import numpy as np
from PIL import Image

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('data/multim_poem.json') as f, open('data/unim_poem.json') as unif:
        multim = json.load(f)
        unim = json.load(unif)

    with open('data/poem_features.pkl', 'rb') as f:
        poem_features = pickle.load(f)

    with open('data/img_features.pkl', 'rb') as f:
        img_features = pickle.load(f)

    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)

    print('vocab size:', len(word2idx))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    basic_tokenizer = BasicTokenizer()

    model = BertGenerator(len(word2idx))
    model = DataParallel(model)
    model.load_state_dict(torch.load(args.load))
    model.to(device)
    model.eval()

    encoder = PoemImageEmbedModel(device)
    encoder = DataParallel(encoder)
    encoder.load_state_dict(torch.load('saved_model/embedder.pth'))
    encoder = encoder.module.img_embedder.to(device)

    examples = [img_features[0], img_features[1], img_features[2],img_features[8], poem_features[0]]

    for feature in examples:
        feature = torch.tensor(feature)
        feature = feature.unsqueeze(0).to(device)
        pred_words = model.module.generate(feature, 70, basic_tokenizer, tokenizer,
                                           word2idx, idx2word, 200, device, args.temp)
        print(' '.join(pred_words).replace(';', '\n'))
        print()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_images = glob2.glob('data/test_image_random/*.jpg')
    test_images.sort()
    for test_image in test_images:
        print('img', test_image)
        pred_words = util.generate_from_one_img_bert(test_image, device, encoder, model,
                                   basic_tokenizer, tokenizer, word2idx, idx2word, args.temp)
        print(' '.join(pred_words).replace(';', ';\n'))
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, default='data/vocab_bert.pkl', help='path for vocabulary file')
    parser.add_argument('--poem-path', type=str, default='data/unim_poem.json', help='path for train poem json file')
    parser.add_argument('-t', '--temp', type=float, default=1)

    args = parser.parse_args()

    main(args)