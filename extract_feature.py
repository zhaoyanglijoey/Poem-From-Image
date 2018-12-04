import json
import argparse
from model import PoemImageEmbedModel
import torch
from torch.nn import DataParallel
from dataloader import convert_to_bert_ids
from pytorch_pretrained_bert import BertTokenizer
import pickle
import torchvision.transforms as transforms
import os, util
from PIL import Image
from tqdm import tqdm

def extract_poem_feature():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    with open('data/unim_poem.json') as unif:
        unim = json.load(unif)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = PoemImageEmbedModel(device)
    encoder = DataParallel(encoder)
    encoder.load_state_dict(torch.load('saved_model/embedder.pth'))
    encoder = encoder.module.poem_embedder.to(device)
    features = {}
    with torch.no_grad():
        for entry in tqdm(unim):
            ids, mask = convert_to_bert_ids(entry['poem'], bert_tokenizer, max_seq_len=100)
            ids = ids.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            poem_embed = encoder(ids, mask)
            poem_embed = poem_embed.cpu().squeeze(0).numpy()
            assert len(poem_embed) == 512
            features[entry['id']] = poem_embed
    with open('data/poem_features.pkl', 'wb') as f:
        pickle.dump(features, f)


def extract_img_feature():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('data/multim_poem.json') as unif:
        multim = json.load(unif)

    multim = util.filter_multim(multim)

    encoder = PoemImageEmbedModel(device)
    encoder = DataParallel(encoder)
    encoder.load_state_dict(torch.load('saved_model/embedder.pth'))
    encoder = encoder.module.img_embedder.to(device)
    features = {}
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img_dir = 'data/image/'

    with torch.no_grad():
        for entry in tqdm(multim):
            img = Image.open(os.path.join(img_dir, '{}.jpg'.format(entry['id']))).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            img_embed = encoder(img)
            img_embed = img_embed.cpu().squeeze(0).numpy()
            assert len(img_embed) == 512
            features[entry['id']] = img_embed
    with open('data/img_features.pkl', 'wb') as f:
        pickle.dump(features, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', '--source', default='poem', help='source to extract; candidate: "poem" or "img", default "poem"')
    args = argparser.parse_args()

    if args.source == 'poem':
        extract_poem_feature()
    elif args.source == 'img':
        extract_img_feature()
    else:
        print('Error: source must be poem or img!')
