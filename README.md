Full TensorFlow MNIST Network
=============================

There were two major things I wanted to accomplish with this network:
1. ETL MNIST data on my own instead of using the tf.learn.examples or some other pre-packaged loader
2. Utilize TensorFlow Datasets instead of using a feed dict

Running `network.py` will first check to see if you have the .gz files
downloaded to the necessary file path. If the files are found, the data is loaded
into numpy arrays and then zipped into tensorflow datasets. The iterators are made,
some basic TensorFlow setup happens, and the network begins training.
