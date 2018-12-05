import argparse
import os
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(image_dir, output_dir, image, size):
    """Resize an image to the given size."""
    with open(os.path.join(image_dir, image), 'r+b') as f:
        with Image.open(f) as img:
            img = img.resize(size, Image.ANTIALIAS)
            img.save(os.path.join(output_dir, image), img.format)

pool = ThreadPool(6)

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]

    images = os.listdir(image_dir)
    pool.map(lambda image: resize_image(image_dir, output_dir, image, image_size), images)
    # resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
