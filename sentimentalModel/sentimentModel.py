import csv
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

image_folder    = 'images1/'
image_list_file = 'image_list.csv'

BATCH_SIZE = 64
TRAIN_COUNT = 2560
VALIDATE_COUNT = 256
IMAGE_SIZE = 224

NUM_EPOCH = 8
preprocessing = False

FC_IN_SIZE = 2048
FC_OUT1_SIZE = 1024
FC_OUT2_SIZE = 128
FC_OUT3_SIZE = 2


learning_rate = 0.005

device = 'cuda' if torch.cuda.is_available() else 'cpu'


resnet50 = torchvision.models.resnet50(pretrained=True)

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def load_vocab():
    vocab = {}
    with open("emotion_vocab.txt", 'r') as vocab_file:
        for line in vocab_file:
            line = line[:-1] # drop last '\n'
            vocab[line] = len(vocab)
    return vocab

emotion_vocab = load_vocab()

class SentimentDataset(Dataset):
    def __init__(self, image_list, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform
        # self.resnet = ResNet50Backbone()
        if device is 'cuda':
            resnet.cuda()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        vocab_len = len(emotion_vocab)
        image_info = self.image_list[index]
        image_filename = image_info[0]
        image_tag = torch.zeros(2)
        # image_tag = torch.zeros(vocab_len + 2)
        emotion = image_info[1]
        # image_tag[emotion_vocab[emotion]] = 1
        # image_tag[vocab_len] = float(image_info[2])
        # image_tag[vocab_len + 1] = float(image_info[3])
        image_tag[0] = float(image_info[2])
        image_tag[1] = float(image_info[3])
        img = torch.tensor([0])
        if preprocessing:
            img = Image.open(self.image_dir + image_filename).convert('RGB')
            img = self.transform(img)

        if device is 'cuda':
            img = img.cuda()
            image_tag = image_tag.cuda()
        return self.image_dir + image_filename, img, image_tag

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super(ResNet50Backbone, self).__init__()
        pretrained_resnet = models.resnet50(pretrained=True)
        self.resnet50_backbone = nn.Sequential(*list(pretrained_resnet.children())[:-1])

    def forward(self, img):
        out = self.resnet50_backbone(img)
        out = out.view(out.size(0), -1)
        return out

class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(FC_IN_SIZE, FC_OUT1_SIZE)
        self.fc2 = nn.Linear(FC_OUT1_SIZE, FC_OUT2_SIZE)
        self.fc3 = nn.Linear(FC_OUT2_SIZE, FC_OUT3_SIZE)

    def forward(self, img_out):
        out = self.fc1(img_out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class SentimentModelWithResNet(nn.Module):
    def __init__(self):
        super(SentimentModelWithResNet, self).__init__()
        self.resnet = ResNet50Backbone()
        self.fc1 = nn.Linear(FC_IN_SIZE, FC_OUT1_SIZE)
        self.fc2 = nn.Linear(FC_OUT1_SIZE, FC_OUT2_SIZE)
        self.fc3 = nn.Linear(FC_OUT2_SIZE, FC_OUT3_SIZE)
    def forward(self, img):
        out = self.resnet(img)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

def preprocess_resnet(dataset_loader):
    preprocessed_dataset = []
    for image_file, image, img_tag in tqdm(dataset_loader):
        # print(image_file)
        tensor_file = '/tmp/' + image_file[0] + '.pt'
        if image is not None:
            if preprocessing:
                resnet_out = resnet(image)
                torch.save(resnet_out, tensor_file)
            # tag = tag.view(1, 1)
            # img_tag = img_tag.float()
            preprocessed_dataset.append((tensor_file, img_tag))
    return preprocessed_dataset


if __name__ == '__main__':

    with open(image_list_file, 'r') as file:
        reader = csv.reader(file)
        image_info_list = list(reader)

    load_vocab()
    resnet = ResNet50Backbone()
    # model = SentimentModelWithResNet()
    model = SentimentModel()

    if device is 'cuda':
        resnet.cuda()
        model.cuda()

    train_image_list = image_info_list[:TRAIN_COUNT]
    valid_image_list = image_info_list[TRAIN_COUNT:TRAIN_COUNT + VALIDATE_COUNT]

    train_dataset = SentimentDataset(train_image_list, image_folder, image_transform)
    validate_dataset = SentimentDataset(valid_image_list, image_folder, image_transform)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
    valid_loader = DataLoader(validate_dataset, batch_size=1, num_workers=0)


    print("Preprocessing images")
    preprocessed_training_set = preprocess_resnet(train_loader)
    preprocessed_validate_set = preprocess_resnet(valid_loader)
    # preprocessed_training_set = train_loader
    # preprocessed_validate_set = valid_loader

    loss_criterion = nn.MSELoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) +
                          list(model.fc2.parameters()) +
                          list(model.fc3.parameters()),
                           lr=learning_rate, momentum=0.9)

    print("start training... ")
    for epoch in range(NUM_EPOCH):
        running_loss = 0.0
        for img_feature_file, tag in tqdm(preprocessed_training_set):
            optimizer.zero_grad()
            img_feature = torch.load(img_feature_file)
            output = model(img_feature)
            loss = loss_criterion(output, tag)
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
        print("epoch {} training finished, training error = {}".format(epoch, running_loss))

        with torch.no_grad():
            running_loss = 0.0
            for img_feature_file, tag in tqdm(preprocessed_validate_set):
                img_feature = torch.load(img_feature_file)
                output = model(img_feature)
                loss = loss_criterion(output, tag)
                running_loss += loss.item()
            print("epoch {} validation finished, validation error = {}".format(epoch, running_loss))

        saved_sate_dict = model.state_dict()
        save_file_name = "./models/training_epoch{}.model".format(epoch)
        torch.save(saved_sate_dict, save_file_name)

