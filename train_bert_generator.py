import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer, BertModel, BertForMaskedLM
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
    if args.pt:
        unim = unim[:2000]

    with open('data/poem_features.pkl', 'rb') as f:
        features = pickle.load(f)

    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    basic_tokenizer = BasicTokenizer()
    unim_dataset = dataloader.UnimDataset(unim, features, basic_tokenizer, tokenizer, word2idx, max_seq_len=100)
    unim_dataloader = DataLoader(unim_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = BertLMGenerator(len(word2idx))
    model = DataParallel(model)
    if args.restore:
        model.load_state_dict(torch.load(args.ckpt))
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=0.33)

    sys.stderr.write('Start training...\n')
    total_step = len(unim_dataloader)

    for epoch in range(args.num_epochs):
        scheduler.step()
        running_ls = 0
        acc_ls = 0
        start = time.time()
        for i, (batch) in enumerate(unim_dataloader):
            id, attn_mask, align_mask, word_ind, lengths_m1, feature = [t.to(device) for t in batch]
            # outputs = model(id, attn_mask, align_mask, feature)
            # # aligned_outputs = outputs[align_mask == 1]
            #
            # word_ind[:, 0] = 0
            # targets = word_ind[word_ind!=0]
            # loss = criterion(outputs, targets)
            targets = id.clone()
            targets[:, 0] = 0
            targets = torch.cat([targets[:, 1:], targets[:, 0].unsqueeze(1) ], dim=1)
            targets[targets==0] = -1
            loss = model(id, attn_mask, targets)
            loss.backward(torch.ones_like(loss))
            running_ls += loss.mean().item()
            acc_ls += loss.mean().item()

            # if (i+1) % args.update_step == 0:
            optimizer.step()
            model.zero_grad()

            if (i+1) % args.log_step == 0:
                elapsed_time = time.time() - start
                iters_per_sec = (i + 1) / elapsed_time
                remaining = (total_step - i - 1) / iters_per_sec
                remaining_fmt = time.strftime("%H:%M:%S", time.gmtime(remaining))
                elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                print('[{}/{}, {}/{}], ls: {:.2f}, Acc: {:.2f} Perplexity: {:5.2f} {:.3}it/s {}<{}'
                      .format(epoch+1, args.num_epochs, i+1, total_step, running_ls / args.log_step,
                              acc_ls / (i+1), np.exp(acc_ls / (i+1)),
                              iters_per_sec, elapsed_fmt, remaining_fmt ) )
                running_ls = 0

            if (i+1) % args.save_step == 0:
                torch.save(model.state_dict(), args.ckpt)

    torch.save(model.state_dict(), args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='saved_model/bertlm_generator.pth', help='path for saving trained models')
    parser.add_argument('--load', type=str)
    parser.add_argument('--ckpt', default='saved_model/bertlm_generator_ckpt.pth')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary file')
    parser.add_argument('--poem-path', type=str, default='data/unim_poem.json', help='path for train poem json file')
    parser.add_argument('--log-step', type=int, default=50, help='step size for prining log info')
    parser.add_argument('--save-step', type=int, default=50, help='step size for saving trained models')
    parser.add_argument('--update-step', type=int, default=16, help='update weights step')
    parser.add_argument('--embed-size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden-size', type=int, default=512, help='dimension of lstm hidden states')

    parser.add_argument('-e', '--num-epochs', type=int, default=10)

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--pt', default=False, action='store_true', help='prototype mode')
    parser.add_argument('-r', '--restore', default=False, action='store_true', help='restore from check point')
    args = parser.parse_args()

    main(args)