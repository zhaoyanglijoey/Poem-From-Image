import json
from tqdm import tqdm
import sys, os

def build_vocab(data):
    sys.stderr.write('building vocab...\n')
    word2idx = {}
    idx2word = {}
    word2idx['<SOS>'] = 0
    word2idx['<EOS>'] = 1
    idx2word[0] = '<SOS>'
    idx2word[1] = '<EOS>'
    for entry in data:
        poem = entry['poem'].replace('\n', ' \n ').split(' ')
        for word in poem:
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word
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
