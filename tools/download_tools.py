#!/usr/bin/python3

import os
from tqdm import tqdm
from urllib.request import urlretrieve

EASY_FEATURES_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
EASY_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
HARD_FEATURES_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
HARD_LABELS_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def download_mnist(url, filename):
    download_folder = './tools/data'
    save_path = os.path.join(download_folder, filename)

    # Create the data folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Download the files if you don't already have them
    if os.path.exists(save_path):
        print('{} data found at location: {}'.format(\
            filename, os.path.abspath(save_path)))
        return
    else:
        with TQDM_Handler(unit='B', unit_scale=True, miniters=1,
                          desc='Downloading {}'.format(filename)) as bar:
            urlretrieve(url, save_path, bar.report_hook)

def setup():
    download_mnist(EASY_FEATURES_URL, 'easy_features.gz')
    download_mnist(EASY_LABELS_URL, 'easy_labels.gz')
    download_mnist(HARD_FEATURES_URL, 'hard_features.gz')
    download_mnist(HARD_LABELS_URL, 'hard_labels.gz')


class TQDM_Handler(tqdm):
    last_block = 0
    def report_hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
