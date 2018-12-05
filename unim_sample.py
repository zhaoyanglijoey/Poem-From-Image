import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.nn import DataParallel

import util
from model import PoemImageEmbedModel
from generative_network.model import DecoderRNN
import dataloader
import json, pickle

def sample_from_poem(poem, encoder, decoder, bert_tokenizer, bert_max_seq_len, idx2word, device):
    ids, mask = dataloader.convert_to_bert_ids(poem, bert_tokenizer, bert_max_seq_len)
    ids = ids.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    # encode
    poem_embed = encoder(ids, mask)
    # decode
    result = []
    samped_indices = decoder.sample(poem_embed)
    samped_indices = samped_indices.cpu().numpy()[0]
    print(samped_indices)
    for word_idx in samped_indices:
        word = idx2word[word_idx]
        if word == '.':
            word = '\n'
        elif word == '<EOS>':
            break
        result.append(word)

    return " ".join(result)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # make sure vocab exists
    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)
    # will be used in embedder
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_max_seq_len = 100

    # # create data loader. the data will be in decreasing order of length
    # data_loader = get_poem_poem_dataset(args.batch_size, shuffle=True, num_workers=args.num_workers, json_obj=unim,
    #                                     max_seq_len=bert_max_seq_len, word2idx=word2idx, tokenizer=bert_tokenizer)

    # init encode & decode model
    # encoder = PoemImageEmbedModel(device)
    # encoder = DataParallel(encoder)
    # encoder.load_state_dict(torch.load(args.encoder_path))
    # encoder = encoder.module.poem_embedder.to(device)
    # encoder = DataParallel(encoder)

    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(word2idx), device).to(device)
    decoder = DataParallel(decoder)
    decoder.load_state_dict(torch.load(args.decoder_path))
    decoder = decoder.module.to(device)
    decoder.eval()

    with open('data/multim_poem.json') as f, open('data/unim_poem.json') as unif:
        multim = json.load(f)
        unim = json.load(unif)

    with open('data/poem_features.pkl', 'rb') as f:
        poem_features = pickle.load(f)

    with open('data/img_features.pkl', 'rb') as f:
        img_features = pickle.load(f)

    word2idx, idx2word = util.read_vocab_pickle(args.vocab_path)

    examples = [poem_features[0], img_features[0]]
    for feature in examples:
        feature = torch.tensor(feature).unsqeeze(0).to(device)
        sample_ids = decoder.sample(feature).cpu().numpy()[0]
        result = []
        for word_idx in sample_ids:
            word = idx2word[word_idx]
            if word == ';':
                word = ';\n'
            elif word == '<EOS>':
                break
            result.append(word)
        print(" ".join(result))

    # poem = args.poem
    # result = sample_from_poem(poem, encoder, decoder, bert_tokenizer, bert_max_seq_len, idx2word, device)
    # print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poem', type=str, required=True, help='input poem for generating poem')
    parser.add_argument('--encoder-path', type=str, default='saved_model/embedder.pth',
                        help='path for trained encoder')
    parser.add_argument('--decoder-path', type=str, default='saved_model/decoder-1.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed-size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--hidden-size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    # print(args.poem)
    main(args)
