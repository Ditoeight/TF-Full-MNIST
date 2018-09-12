import sys

import tensorflow as tf
import numpy as np

from tools.etl_tools import load_mnist_features, load_mnist_labels
from tools.download_tools import setup

# Parameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
HIDDEN_NODES = 800
TRAIN_DATA_SPLIT = 0.71429 # Percentage of data used for training

################################################################################
#                     Do not edit anything below this line                     #
################################################################################

# Validate data is downloaded
setup()

# Load datasets and create iterators
x_train, x_valid, x_test = load_mnist_features(TRAIN_DATA_SPLIT)
y_train, y_valid, y_test = load_mnist_labels(TRAIN_DATA_SPLIT)

train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .batch(BATCH_SIZE).repeat()
valid_set = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))\
    .batch(BATCH_SIZE).repeat()
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
    .batch(BATCH_SIZE).repeat()

iterator = tf.data.Iterator.from_structure(train_set.output_types,
                                           train_set.output_shapes)

next_batch = iterator.get_next()

train_init = iterator.make_initializer(train_set)
valid_init = iterator.make_initializer(valid_set)
test_init = iterator.make_initializer(test_set)

# Define the network
def network(inputs):
    dropout = tf.layers.dropout(
        inputs,
        rate=0.4
    )

    batch_normalization = tf.layers.batch_normalization(
        dropout
    )

    hidden_1 = tf.layers.dense(
        batch_normalization,
        HIDDEN_NODES,
        activation=tf.sigmoid,
        kernel_initializer=tf.initializers.truncated_normal
    )

    output = tf.layers.dense(
        hidden_1,
        10
    )

    return output

# Build Model
logits = network(next_batch[0])

#        |  ||
# Define || |_ and Optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
    logits=logits, labels=next_batch[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)\
    .minimize(loss)

# Check accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(next_batch[1], 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):

        sess.run(train_init) # Training run
        for batch in range(x_train.shape[0]//BATCH_SIZE):
            sess.run(optimizer)

        sess.run(valid_init) # Validation run
        valid_acc = 0
        for batch in range(x_valid.shape[0]//BATCH_SIZE):
            a = sess.run(accuracy)
            valid_acc += a
        print('Epoch {:4d} '.format(epoch + 1)\
              + 'validation accuracy: {:.2f}%'.format(
                (valid_acc/(x_valid.shape[0]//BATCH_SIZE)) * 100)
        )

    sess.run(test_init) # Test run
    test_acc = 0
    for batch in range(x_test.shape[0]//BATCH_SIZE):
        a = sess.run(accuracy)
        test_acc += a
    print('Test accuracy: {:.4f}%'.format(
        (test_acc/(x_test.shape[0]//BATCH_SIZE)) * 100)
    )
