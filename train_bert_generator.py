import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer, BertModel
import os, sys, time
import dataloader
from model import VGG16_fc7_object, PoemImageEmbedModel, BertGenerator
import json, pickle
from util import load_vocab_json, build_vocab
from torch.nn.utils.rnn import pack_padded_sequence
from generative_network.model import DecoderRNN
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
        features = pickle.load(f)

    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    basic_tokenizer = BasicTokenizer()
    unim_dataset = dataloader.build_unim_dataset(unim, features, basic_tokenizer, tokenizer, word2idx, max_seq_len=256)
    model = BertGenerator(len(word2idx))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.33)

    sys.stderr.write('Start training...\n')
    total_step = len(unim_dataset)

    for epoch in range(args.num_epochs):
        scheduler.step()
        for i, (batch) in enumerate(tqdm(unim_dataset)):
            id, attn_mask, align_mask, word_ind = [t.to(device) for t in batch]
            outputs = model(id, attn_mask)[0]
            aligned_outputs = outputs[align_mask][:-1]
            targets = word_ind[0][1:]
            loss = criterion(aligned_outputs, targets)
            loss.backward()

            if (i+1) % args.update_step == 0:
                optimizer.step()
                model.zero_grad()

            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='saved_model/embedder.pth' , help='path for loading pre-trained models')
    parser.add_argument('--save-model-path', type=str, default='saved_model' , help='path for saving trained models')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary file')
    parser.add_argument('--poem-path', type=str, default='data/unim_poem.json', help='path for train poem json file')
    parser.add_argument('--log-step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save-step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--update-step', type=int, default=16, help='update weights step')
    parser.add_argument('--embed-size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden-size', type=int, default=512, help='dimension of lstm hidden states')

    parser.add_argument('--num_epochs', type=int, default=5)

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    main(args)