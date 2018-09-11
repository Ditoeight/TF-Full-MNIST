import gzip
import numpy as np
import tensorflow as tf

from .download_tools import setup

def read_bytestream(bytestream):
    """Read 32 bit integers from the bytestream

    The MNIST .gz files are capped with qualitative information
    represented by 32 bit integers stored in high endian format, so this
    function is to help read and flip each cap.

    More information can be found at yann.lecun.com/exdb/mnist
    """
    high_endian = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=high_endian)[0]

def extract_mnist_features(file):
    """
    Extract features from the .gz file
    """
    with gzip.open(file) as bytestream:
        bytestream.read(4) # Ditch the magic number
        records = read_bytestream(bytestream)
        height = read_bytestream(bytestream)
        width = read_bytestream(bytestream)
        buffer = bytestream.read(records * height * width)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(records, 28 * 28)
        return data

def extract_mnist_labels(file):
    """
    Extract labels from the .gz file
    """
    with gzip.open(file) as bytestream:
        bytestream.read(4) # Ditch the magic number
        records = read_bytestream(bytestream)
        buffer = bytestream.read(records)
        data = np.frombuffer(buffer, dtype=np.uint8)
        return data

def transform_mnist_features():
    """
    Combine and transform the MNIST features to a range between 0
    and 1
    """
    easy_features = extract_mnist_features('tools/data/easy_features.gz')
    hard_features = extract_mnist_features('tools/data/hard_features.gz')
    features = np.concatenate(\
        (easy_features, hard_features)).astype('float32') / 255

    return features

def transform_mnist_labels(onehot=True):
    """
    Combine and optionally transform labels to one-hot encoding
    """
    easy_labels = extract_mnist_labels('tools/data/easy_labels.gz')
    hard_labels = extract_mnist_labels('tools/data/hard_labels.gz')
    labels = np.concatenate((easy_labels, hard_labels))
    if not onehot:
        return labels
    labels = np.eye(10)[labels]

    return labels

def split_data(data, train_pct, validation):
    """Split the data into train, test, and validation sets

    The number of records in each set is determined by the percentage of
    data saved for training. The remaining amount is split evenly
    between testing and validation sets.

    For example, setting a train_pct of .8 will save 80% of the data for
    training, and then split the remaining 20% in half for 10% of
    training and 10% of validation data.
    """
    if validation:
        split_1 = int(data.shape[0] * train_pct)
        split_2 = int(data.shape[0] * ((1 - train_pct)/2)) + split_1

        train, test, valid = np.split(data, [split_1, split_2])

        return train, test, valid
    else:
        split = int(data.shape[0] * train_pct)
        train, test = np.split(data, [split])

        return train, x_test

def load_mnist_features(train_pct, validation=True):
    """Load features to np arrays
    """
    features = transform_mnist_features()
    return split_data(features, train_pct, validation)

def load_mnist_labels(train_pct, onehot=True, validation=True):
    """Load labels to np arrays
    """
    labels = transform_mnist_labels(onehot=onehot)
    return split_data(labels, train_pct, validation)
