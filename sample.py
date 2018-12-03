import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from generative_network.model import GRUDecoder


def poem_from_image(img: Image, encoder, decoder: GRUDecoder) -> str:
    # TODO: image preprocess

    # TODO: get image features from encoder (3 CNNs)
    features = None

    # TODO: decode features and generate words indices
    indices = decoder.sample(features)
    # TODO: indices to words
    poem = ""

    return poem


def main(args):
    img = Image.open(args.image)
    # TODO: init vocab dict

    # TODO: init encoder (CNN)
    encoder = None

    # TODO: init decoder (GRU)
    decoder = GRUDecoder(args.hidden_size, args.vocab_siz)

    poem =(img, encoder, decoder)
    plt.imshow(np.asarray(img))
    print(poem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-1000.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-1000.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    args = parser.parse_args()

    main(args)
