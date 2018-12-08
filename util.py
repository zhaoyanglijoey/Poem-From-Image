import json
from tqdm import tqdm
import sys, os
import collections
import nltk
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

def generate_from_one_img_lstm(test_image, device, encoder, decoder,temp):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(test_image).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(device)
    img_embed = encoder(img)
    pred_words = decoder.module.sample(img_embed, temperature=temp).cpu().numpy()[0]
    return pred_words

def generate_from_one_img_bert(test_image, device, encoder, decoder,
                     basic_tokenizer, tokenizer,word2idx, idx2word, temp):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(test_image).convert('RGB')
    img = test_transform(img).unsqueeze(0).to(device)
    img_embed = encoder(img)
    pred_words = decoder.module.generate(img_embed, 70, basic_tokenizer, tokenizer,
                                       word2idx, idx2word, 200, device, temp)
    return pred_words

def normalize(t):
    out = t / torch.norm(t, dim=-1, keepdim=True)
    return out

def add_word(word2idx, idx2word, word):
    if word not in word2idx:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word


def process_one_poem(poem):
    """
    from string to tokens
    :param poem: poem string
    :return: list: tokens
    """

    poem = poem.replace('\n', ' ; ')  # use "." as new line symbol?
    tokens = nltk.tokenize.word_tokenize(poem)
    return tokens


def build_vocab(data, threshold):
    counter = collections.Counter()
    sys.stderr.write('building vocab...\n')
    word2idx = {}
    idx2word = {}
    add_word(word2idx, idx2word, '<PAD>')  # padding
    add_word(word2idx, idx2word, '<SOS>')  # start of poem
    add_word(word2idx, idx2word, '<EOS>')  # end of sentence (end of poem)
    # add_word(word2idx, idx2word, '<EOL>')  # end of line
    add_word(word2idx, idx2word, '<UNK>')  # known

    sys.stderr.write('Parsing data...\n')
    for entry in tqdm(data):
        tokens = process_one_poem(entry['poem'])
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    sys.stderr.write('Adding words...\n')
    for word in tqdm(words):
        add_word(word2idx, idx2word, word)

    return word2idx, idx2word

def build_vocab_bert(data, threshold):
    counter = collections.Counter()
    sys.stderr.write('building vocab...\n')
    word2idx = {}
    idx2word = {}
    add_word(word2idx, idx2word, '[PAD]')  # padding
    add_word(word2idx, idx2word, '[CLS]')  # start of poem
    add_word(word2idx, idx2word, '[SEP]')  # end of sentence (end of poem)
    # add_word(word2idx, idx2word, '<EOL>')  # end of line
    add_word(word2idx, idx2word, '[UNK]')  # known

    basic_tokenizer = BasicTokenizer()

    sys.stderr.write('Parsing data...\n')
    for entry in tqdm(data):
        # tokens = process_one_poem(entry['poem'])
        tokens = basic_tokenizer.tokenize(entry['poem'].replace('\n', ' ; '))
        # tokens = entry['poem'].replace('\n', ' ; ').split()
        counter.update(tokens)
        # [add_word(word2idx, idx2word, word) for word in tokens]

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


def filter_sentiment(df, img_dir):
    imgs = os.listdir(img_dir)
    valid_ids = set()
    for img in imgs:
        valid_ids.add(int(os.path.splitext(img)[0]))

    def valid(id):
        return True if id in valid_ids else False

    is_valid = df.id.apply(valid)
    ret = df[is_valid == True]
    return ret
