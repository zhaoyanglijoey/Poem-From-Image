import json
from tqdm import tqdm
import sys, os
import collections
import nltk
import pickle


def add_word(word2idx, idx2word, word):
    if word not in word2idx:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word


def build_vocab(data, threshold):
    counter = collections.Counter()
    sys.stderr.write('building vocab...\n')
    word2idx = {}
    idx2word = {}
    add_word(word2idx, idx2word, '<PAD>')  # padding
    add_word(word2idx, idx2word, '<SOS>')  # start of poem
    # add_word(word2idx, idx2word, '<EOS>')  # end of sentence (end of poem)
    add_word(word2idx, idx2word, '<EOL>')  # end of line
    add_word(word2idx, idx2word, '<UNK>')  # known

    sys.stderr.write('Parsing data...\n')
    for entry in tqdm(data):
        poem = entry['poem'].replace('\n', ' . ')  # use "." as new line symbol?
        tokens = nltk.tokenize.word_tokenize(poem)
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    sys.stderr.write('Adding words...\n')
    for word in tqdm(words):
        add_word(word2idx, idx2word, word)

    return word2idx, idx2word


def read_vocab_pickle(file):
    if not os.path.exists(file):
        raise FileNotFoundError("Not found vocab files. Please run `python vocab_builder.py` first")
    with open(file, 'rb') as f:
        word2idx, idx2word = pickle.load(f)
    return word2idx, idx2word


def load_vocab_json(file):
    with open(file) as f:
        word2idx = json.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def filter_multim(multim):
    output = []
    imgs = os.listdir('data/image/')
    valid_ids = set()
    for img in imgs:
        valid_ids.add(int(os.path.splitext(img)[0]))
    for entry in multim:
        if entry['id'] in valid_ids:
            output.append(entry)
    return output
