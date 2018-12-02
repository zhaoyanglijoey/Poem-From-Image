import csv, os
import random
from tqdm import tqdm
from urllib import request

download_max = 128

source_folder = './agg/'
test_csv_file = './agg/anger_1st_1000.csv'

image_save_folder = './images/'
image_info_file = './image_list.csv'
images_with_tag = []

def build_emotion_vocab(image_list):
    vocab = {}
    f = open("emotion_vocab.txt", 'w')
    for image_info in image_list:
        emotion = image_info[0]
        if emotion not in vocab:
            vocab[emotion] = len(vocab)
            print(emotion, file=f)
            print(emotion)
    f.close()


def download_image(imageURL):
    response = request.urlopen(imageURL)
    if imageURL == response.geturl():
        filename = imageURL.split('/')[-1]
        with open(image_save_folder + filename, 'wb') as f:
            f.write(response.read())
        return filename
    else:
        return ''

def readCSVList(csvFile):
    with open(csvFile, 'r') as file:
        reader = csv.reader(file)
        image_list = list(reader)
    return image_list

def download_image_list(image_list):
    print("now downloading {} images".format(len(image_list)))

    dump_info_file = open(image_info_file, 'w')
    for image_info in tqdm(image_list):
        image_file = download_image(image_info[1])
        if image_file:
            # images_with_tag.append((image_file, image_info[2]))
            print(image_file, image_info[0], image_info[2], image_info[3], file=dump_info_file, sep=',')
    dump_info_file.close()


def download_all_images():
    all_image_list = []
    for csvFile in os.listdir(source_folder):
        all_image_list += readCSVList(source_folder + csvFile)

    build_emotion_vocab(all_image_list)
    random.shuffle(all_image_list)
    all_image_list = all_image_list[:download_max]
    download_image_list(all_image_list)

if __name__ == '__main__':
    download_all_images()
