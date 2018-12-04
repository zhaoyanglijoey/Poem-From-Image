import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer, BertModel
import os, sys, time
import dataloader
from model import BertGenerator, BertLMGenerator
import json, pickle
from tqdm import tqdm
import argparse
import util
import numpy as np

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

    model = BertLMGenerator(len(word2idx))
    model = DataParallel(model)
    model.load_state_dict(torch.load(args.load))
    model.to(device)

    examples = [poem_features[0], img_features[0]]

    for feature in examples:
        feature = torch.tensor(feature)
        pred_words = model.module.generate(feature, 70, basic_tokenizer, tokenizer, word2idx, idx2word, 200, device)
        # pred_words = pred_words[:-1]
        print(' '.join(pred_words))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary file')
    parser.add_argument('--poem-path', type=str, default='data/unim_poem.json', help='path for train poem json file')
    args = parser.parse_args()

    main(args)