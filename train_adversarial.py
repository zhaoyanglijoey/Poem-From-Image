import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import os, sys, time
from torch.nn.utils.rnn import pad_packed_sequence
from dataloader import PoemImageDataset, PoemImageEmbedDataset, get_poem_poem_dataset
from model import PoemImageEmbedModel, DecoderRNN, Discriminator
import json, pickle
from util import load_vocab_json, build_vocab
from torch.nn.utils.rnn import pack_padded_sequence
# from generative_network.model import DecoderRNN
from tqdm import tqdm
import argparse
import util
import numpy as np


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('data/multim_poem.json') as f, open('data/unim_poem.json') as unif:
        multim = json.load(f)
        unim = json.load(unif)

    multim = util.filter_multim(multim)
    # multim = multim[:128]
    with open('data/img_features.pkl', 'rb') as fi, open('data/poem_features.pkl', 'rb') as fp:
        img_features = pickle.load(fi)
        poem_features = pickle.load(fp)


    # make sure vocab exists
    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)

    # will be used in embedder

    if args.source == 'unim':
        data = unim
        features = poem_features
    elif args.source == 'multim':
        data = multim
        features = img_features
    else:
        print('Error: source must be unim or multim!')
        exit()

    # create data loader. the data will be in decreasing order of length
    data_loader = get_poem_poem_dataset(args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, json_obj=data, features=features,
                                        max_seq_len=128, word2idx=word2idx, tokenizer=None)

    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(word2idx), device)
    decoder = DataParallel(decoder)
    if args.restore:
        decoder.load_state_dict(torch.load(args.ckpt))
    if args.load:
        decoder.load_state_dict(torch.load(args.load))
    decoder.to(device)

    discriminator = Discriminator(args.embed_size, args.hidden_size, len(word2idx), num_labels=2)
    discriminator.embed.weight = decoder.module.embed.weight
    discriminator = DataParallel(discriminator)
    if args.restore:
        discriminator.load_state_dict(torch.load(args.disc))
    discriminator.to(device)

    # optimization config
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 10], gamma=0.33)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    sys.stderr.write('Start training...\n')
    total_step = len(data_loader)
    decoder.train()
    global_step = 0
    running_ls = 0
    for epoch in range(args.num_epochs):
        scheduler.step()
        acc_ls = 0
        start = time.time()

        for i, (batch) in enumerate(data_loader):
            poem_embed, ids, lengths = [t.to(device) for t in batch]
            targets = pack_padded_sequence(ids[:, 1:], lengths, batch_first=True)[0]
            # train discriminator

            # train with real
            discriminator.zero_grad()
            pred_real = discriminator(ids[:, 1:], lengths, poem_embed)
            real_label = torch.ones(ids.size(0), dtype=torch.long).to(device)
            loss_d_real = criterion(pred_real, real_label)
            loss_d_real.backward(torch.ones_like(loss_d_real), retain_graph=True)

            # train with fake

            logits = decoder(poem_embed, ids, lengths)
            weights = F.softmax(logits, dim=-1)
            m = Categorical(probs=weights)
            generated_ids = m.sample()

            # generated_ids = torch.argmax(logits, dim=-1)
            pred_fake = discriminator(generated_ids.detach(), lengths, poem_embed)
            fake_label = torch.zeros(ids.size(0)).long().to(device)
            loss_d_fake = criterion(pred_fake, fake_label)
            loss_d_fake.backward(torch.ones_like(loss_d_fake), retain_graph=True)

            loss_d = loss_d_real.mean().item() + loss_d_fake.mean().item()

            optimizerD.step()

            # train generator
            decoder.zero_grad()
            reward = F.softmax(pred_fake, dim=-1)[:, 1].unsqueeze(-1)
            loss_r = -m.log_prob(generated_ids) * reward
            loss_r.backward(torch.ones_like(loss_r), retain_graph=True)
            loss_r = loss_r.mean().item()

            loss = criterion(pack_padded_sequence(logits, lengths, batch_first=True)[0], targets)
            loss.backward(torch.ones_like(loss))
            loss = loss.mean().item()
            # loss = loss_r
            running_ls += loss
            acc_ls += loss

            for param in decoder.parameters():
                torch.nn.utils.clip_grad_norm_(param, 0.25)

            optimizer.step()
            global_step += 1

            if global_step % args.log_step == 0:
                elapsed_time = time.time() - start
                iters_per_sec = (i + 1) / elapsed_time
                remaining = (total_step - i - 1) / iters_per_sec
                remaining_fmt = time.strftime("%H:%M:%S", time.gmtime(remaining))
                elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                print('[{}/{}, {}/{}], ls_d:{:.2f}, ls_r:{:.2f} ls: {:.2f}, Acc: {:.2f} Perp: {:5.2f} {:.3}it/s {}<{}'
                      .format(epoch+1, args.num_epochs, i+1, total_step, loss_d, loss_r,
                              running_ls / args.log_step, acc_ls / (i+1), np.exp(acc_ls / (i+1)),
                              iters_per_sec, elapsed_fmt, remaining_fmt ) )
                running_ls = 0

            if global_step % args.save_step == 0:
                torch.save(decoder.state_dict(), args.ckpt)
                torch.save(discriminator.state_dict(), args.disc)
    torch.save(decoder.state_dict(), args.save)
    torch.save(discriminator.state_dict(), args.disc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='saved_model/lstm_gen_D_f.pth' , help='path for saving trained models')
    parser.add_argument('--disc', default='saved_model/discriminator_f.pth')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary file')
    parser.add_argument('--log-step', type=int, default=50, help='step size for prining log info')
    parser.add_argument('--save-step', type=int, default=200, help='step size for saving trained models')

    parser.add_argument('--embed-size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden-size', type=int, default=256, help='dimension of lstm hidden states')

    parser.add_argument('-e' ,'--num-epochs', type=int, default=100)

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('-r', '--restore', default=False, action='store_true', help='restore from check point')
    parser.add_argument('--ckpt', default='saved_model/lstm_gen_D_f_ckpt.pth')
    parser.add_argument('--load')
    parser.add_argument('--source', default='unim', help='training data; unim or multim')

    args = parser.parse_args()
    main(args)
