import json
from tqdm import tqdm
import sys


def add_word(word2idx, idx2word, word):
    if word not in word2idx:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word


def build_vocab(data):
    sys.stderr.write('building vocab...\n')
    word2idx = {}
    idx2word = {}
    add_word(word2idx, idx2word, '<PAD>')
    add_word(word2idx, idx2word, '<SOS>')
    add_word(word2idx, idx2word, '<EOS>')
    add_word(word2idx, idx2word, '<UNK>')
    for entry in data:
        poem = entry['poem'].replace('\n', ' \n ').split(' ')
        for word in poem:
            add_word(word2idx, idx2word, word)
    return word2idx, idx2word


def load_vocab_json(file):
    with open(file) as f:
        word2idx = json.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word