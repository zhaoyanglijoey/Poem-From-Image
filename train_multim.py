import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer
import os, sys, time
from dataloader import PoemImageDataset, PoemImageEmbedDataset, get_poem_poem_dataset
from model import VGG16_fc7_object, PoemImageEmbedModel
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

    with open('data/img_features.pkl', 'rb') as f:
        features = pickle.load(f)

    # make sure vocab exists
    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)

    # will be used in embedder
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_max_seq_len = 100

    # create data loader. the data will be in decreasing order of length
    data_loader = get_poem_poem_dataset(args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, json_obj=multim, features=features,
                                        max_seq_len=bert_max_seq_len, word2idx=word2idx, tokenizer=bert_tokenizer)

    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(word2idx), device)
    if args.restore:
        decoder.load_state_dict(torch.load(args.ckpt))
    decoder = DataParallel(decoder)
    decoder.to(device)

    # optimization config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.33)

    sys.stderr.write('Start training...\n')
    total_step = len(data_loader)
    decoder.train()
    for epoch in range(args.num_epochs):
        scheduler.step()
        for i, (batch) in enumerate(tqdm(data_loader)):
            poem_embed, poems, lengths = [t.to(device) for t in batch]
            targets = pack_padded_sequence(poems[:, 1:], lengths, batch_first=True)[0]

            decoder.zero_grad()
            # poem_embed = encoder(ids, mask)
            outputs = decoder(poem_embed, poems, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i+1, total_step, loss.item(), np.exp(loss.item())))

            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), args.ckpt)
        # Save the model checkpoints
        torch.save(decoder.state_dict(), os.path.join(
            args.save_model_path, 'decoder_multim-{}.ckpt'.format(epoch+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='saved_model/embedder.pth' , help='path for loading pre-trained models')
    parser.add_argument('--save-model-path', type=str, default='saved_model' , help='path for saving trained models')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary file')
    parser.add_argument('--log-step', type=int, default=20, help='step size for prining log info')
    parser.add_argument('--save-step', type=int, default=100, help='step size for saving trained models')

    parser.add_argument('--embed-size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden-size', type=int, default=512, help='dimension of lstm hidden states')

    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('-r', '--restore', default=False, action='store_true', help='restore from check point')
    parser.add_argument('--ckpt', default='saved_model/lstm_gen_multim_ckpt.pth')

    args = parser.parse_args()
    main(args)
