# TODO: finish vocab builder using nltk.tokenize.word_tokenize. Add <UNK> <PAD> tags
import argparse, sys
import json
import util
import pickle


def main(args):
    with open('data/multim_poem.json') as f, open('data/unim_poem.json') as unif:
        multim = json.load(f)
        unim = json.load(unif)

    word2idx, idx2word = util.build_vocab_bert(unim + multim, args.threshold)
    sys.write('vocab size {}'.format(len(word2idx)))
    with open(args.vocab_path, 'wb') as f:
        pickle.dump([word2idx, idx2word], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)