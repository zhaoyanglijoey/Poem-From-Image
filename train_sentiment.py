import torch
import torch.nn as nn
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer
import os, sys, time, argparse, logging
from dataloader import PoemImageDataset, PoemImageEmbedDataset, VisualSentimentDataset
from model import VGG16_fc7_object, PoemImageEmbedModel, Res50_sentiment
import json
from util import load_vocab_json, build_vocab, check_path, filter_multim, filter_sentiment
from tqdm import tqdm
import pandas as pd
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class VisualSentimentTrainer():
    def __init__(self, train_data, test_data, img_dir, batchsize, load_model, device):
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        self.train_set = VisualSentimentDataset(self.train_data, img_dir,
                                               transform=self.train_transform)
        self.train_loader = DataLoader(self.train_set, batch_size=batchsize, shuffle=True, num_workers=4)

        self.test_set = VisualSentimentDataset(self.test_data, img_dir,
                                              transform=self.test_transform)
        self.test_loader = DataLoader(self.test_set, batch_size=batchsize, num_workers=4)

        self.model = Res50_sentiment()
        self.model = DataParallel(self.model)
        if load_model:
            logger.info('load model from '+ load_model)
            self.model.load_state_dict(torch.load(load_model))
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2, 4], gamma=0.5)

    def train_epoch(self, epoch, log_interval, save_interval, ckpt_file):
        self.model.train()
        running_ls = 0
        acc_ls = 0
        start = time.time()
        num_batches = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            img, label = [t.to(self.device) for t in batch]
            self.model.zero_grad()
            pred = self.model(img)
            loss = self.criterion(pred, label)
            loss.backward(torch.ones_like(loss))
            running_ls += loss.mean().item()
            acc_ls += loss.mean().item()
            self.optimizer.step()

            if (i + 1) % log_interval == 0:
                elapsed_time = time.time() - start
                iters_per_sec = (i + 1) / elapsed_time
                remaining = (num_batches - i - 1) / iters_per_sec
                remaining_fmt = time.strftime("%H:%M:%S", time.gmtime(remaining))
                elapsed_fmt = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                print('[{:>2}, {:>4}/{}] running loss:{:.4} acc loss:{:.4} {:.3}iters/s {}<{}'.format(
                    epoch, (i + 1), num_batches, running_ls / log_interval, acc_ls /(i+1),
                    iters_per_sec, elapsed_fmt, remaining_fmt))
                running_ls = 0

            if (i + 1) % save_interval == 0:
                self.save_model(ckpt_file)

    def test(self):
        self.model.eval()
        batches_count = 0
        data_count = 0
        num_correct = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                batches_count += 1
                img, label = tuple(t.to(self.device) for t in batch)
                data_count += img.shape[0]
                logits = self.model(img).cpu().numpy()
                label = label.cpu().numpy()
                num_correct += np.sum(np.argmax(logits, axis=1) == label)

        accuracy = num_correct / data_count
        print('accuracy: {:.4}%'.format(accuracy * 100))

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load-model', default=None)
    argparser.add_argument('-e', '--num_epoch', type=int, default=5)
    argparser.add_argument('-t', '--test', default=False, action='store_true')
    argparser.add_argument('--pt', default=False, action='store_true', help='prototype mode')
    argparser.add_argument('-b', '--batchsize', type=int, default=32)
    argparser.add_argument('--log-interval', type=int, default=10)
    argparser.add_argument('--save-interval', type=int, default=100)
    argparser.add_argument('-r', '--restore', default=False, action='store_true',
                           help='restore from checkpoint')
    argparser.add_argument('--ckpt', default='saved_model/sentiment_ckpt.pth')
    argparser.add_argument('--save', default='saved_model/sentiment.pth')
    args = argparser.parse_args()

    logging.info('reading data')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainfile = 'data/image-sentiment-polarity-all.csv'
    testfile = 'data/image-sentiment-polarity-test.csv'
    # trainfile = 'data/visual_sentiment_train.csv'
    # testfile = 'data/visual_sentiment_test.csv'

    img_dir = 'data/polarity_image/'
    train_data = pd.read_csv(trainfile, dtype={'id':int})
    test_data = pd.read_csv(testfile, dtype={'id':int})
    train_data = filter_sentiment(train_data, img_dir)
    test_data = filter_sentiment(test_data, img_dir)

    logging.info('number of training data:{}, number of testing data:{}'.
                 format(len(train_data), len(test_data)))

    if args.pt:
        train_data = train_data[:1000]
        test_data = test_data[:100]

    logging.info('building model...')
    load_model = args.load_model
    if args.load_model is None and args.restore and os.path.exists(args.ckpt):
        load_model = args.ckpt
    sentiment_trainer = VisualSentimentTrainer(train_data, test_data, img_dir, args.batchsize, load_model, device)
    check_path('saved_model')
    if args.test:
        sentiment_trainer.test()
    else:
        logging.info('start traning')
        for e in range(args.num_epoch):
            sentiment_trainer.train_epoch(e+1, args.log_interval, args.save_interval, args.ckpt)
            sentiment_trainer.scheduler.step()
            sentiment_trainer.test()
            sentiment_trainer.save_model(args.ckpt)
        sentiment_trainer.save_model(args.save)


if __name__ == '__main__':
    main()
