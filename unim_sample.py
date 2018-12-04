import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.nn import DataParallel

import util
from model import PoemImageEmbedModel
from generative_network.model import DecoderRNN
import dataloader


def sample_from_poem(poem, encoder, decoder, bert_tokenizer, bert_max_seq_len, idx2word, device):
    ids, mask = dataloader.convert_to_bert_ids(poem, bert_tokenizer, bert_max_seq_len)
    ids = ids.to(device)
    mask = mask.to(device)

    # encode
    poem_embed = encoder(ids, mask)
    # decode
    result = []
    samped_indices = decoder.sample(poem_embed)
    samped_indices = samped_indices.cpu().numpy()
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
    encoder = PoemImageEmbedModel(device)
    encoder = DataParallel(encoder)
    encoder.load_state_dict(torch.load(args.encoder_path))
    encoder = encoder.module.poem_embedder.to(device)
    encoder = DataParallel(encoder)

    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(word2idx), device).to(device)
    decoder = DataParallel(decoder)
    decoder.load_state_dict(torch.load(args.decoder_path))

    poem = args.poem
    result = sample_from_poem(poem, encoder, decoder, bert_tokenizer, bert_max_seq_len, idx2word, device)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poem', type=str, required=True, help='input poem for generating poem')
    parser.add_argument('--encoder-path', type=str, default='saved_model/embedder.pth',
                        help='path for trained encoder')
    parser.add_argument('--decoder-path', type=str, default='saved_model/decoder-5.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed-size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden-size', type=int, default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
