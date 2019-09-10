# -*- coding: utf-8 -*-

'''
This notebook builds a bidirectional recurrent network using tensorflow and
trains it on the simple task of predicting the next number in Fibonacci like
sequences.

The bidirectional_dynamic_rnn function is used to construct the bidirectional
recurrent part of the network given two cells.
'''

import random
import pickle
import itertools

from tqdm import trange

from collections import namedtuple

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from tensorflow.nn import bidirectional_dynamic_rnn

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import time

def range_batch(low_or_high, high=None, step=32):
    if high is None:
        high = low_or_high
        low_or_high = 0
    low_iter = range(low_or_high, high, step)
    high_iter = range(low_or_high+step, high, step)
    return itertools.zip_longest(low_iter, high_iter)

def window_default(iterator, window):
    for elem in iterator:
        window = window[1:] + (elem,)
        yield window

def window_iter(iterable, size=2):
    it = iter(iterable)
    window = tuple(itertools.islice(it, size))
    if len(window) == size:
        yield window
    yield from window_default(it, window)

def padded_window_iter(iterable, size=2, default=None):
    it = iter(iterable)
    window = (default,)*size
    yield from window_default(it, window)

def process_packet(packet):
    protocol, byte_no, date, label = packet
    protocol = 0 if protocol == '6' else 1
    label = 0 if label == 'BENIGN' else 1
    return protocol, byte_no, label

def process_sequence(sequence):
    target_protocol, target_bytes, label = sequence[-1]
    new_sequence = []
    for protocol, byte_no, label in sequence[:-1]:
        new_sequence.append((protocol, byte_no))
    return new_sequence, (target_protocol, target_bytes), label

def read_data(filename):
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    sequences = []
    for _, source in data.items():
        for _, destination in source.items():
            sequences.append(destination)

    sequence_windows = []
    sequence_targets = []
    sequence_label = []
    for sequence in sequences:
        processed_sequence = (process_packet(i) for i in sequence)
        for window in window_iter(processed_sequence, size=11):
            features, target_token, target_label = process_sequence(window)
            sequence_windows.append(features)
            sequence_targets.append(target_token)
            sequence_label.append(target_label)
    return (np.array(sequence_windows), np.array(sequence_targets),
            np.array(sequence_label).reshape((-1, 1)))

def get_data_summary(windows, targets):
    print("Number of subsequences: ", len(windows))
    protocol = targets[:, 0]
    tcp_num = sum(protocol)
    tcp_percentage = tcp_num/len(targets)
    print("TCP # and %: {:d} {:.2f}%".format(tcp_num, tcp_percentage))

TUESDAY = "Tuesday-workingHours"
WEDNESDAY = "Wednesday-workingHours"
MONDAY = "Monday-workingHours"
THURSDAY = "Thursday-WorkingHours-Afternoon-Infilteration"


def unique_concatenate(data):
    windows, targets = data
    windows = windows.reshape((-1, 2))
    data_concat = np.concatenate((windows, targets))
    return np.unique(data_concat, axis=0)

begin = time()
with ProcessPoolExecutor() as executor:
    future = executor.map(read_data, [MONDAY, TUESDAY, WEDNESDAY, THURSDAY])


monday_data, tuesday_data, wednesday_data, thursday_data = list(future)
print("Time Elapsed: ", time() - begin)

uniqueM = unique_concatenate(monday_data[:2])

uniqueT = unique_concatenate(tuesday_data[:2])

uniqueW = unique_concatenate(wednesday_data[:2])

uniqueTH = unique_concatenate(thursday_data[:2])

unique_concat = np.concatenate((uniqueM, uniqueT, uniqueW, uniqueTH))
unique_all = np.unique(unique_concat, axis=0)

sequence_windows, sequence_targets, _ = monday_data
test_windows, test_targets, test_labels = wednesday_data

L_UNIQUE = len(unique_all)
print("Monday")
get_data_summary(sequence_windows, sequence_targets)
print("Wednesday")
get_data_summary(test_windows, test_targets)
label_an = np.sum(test_labels)
label_an_percent = label_an/len(test_labels)
print("True anomolies: {:d}, {:.2f}%".format(label_an, label_an_percent))

token_dict = {tuple(token): i for i, token in enumerate(unique_all)}
sequence_targets = np.array([token_dict[tuple(target)] for target in sequence_targets], dtype=np.int32)
test_targets = np.array([token_dict[tuple(target)] for target in test_targets], dtype=np.int32)

print("Unique tokens in dataset: ", len(set(sequence_targets)))

def create_lstm(batch_size, num_units, activation=tf.nn.tanh):
    '''Create an lstm cell with a initialization zero matrix.
    
    Creates an LSTM cell by using tensorflow function LSTMCell which creates
    the cell.
    
    i.e. state_is_tuple is used to prevent to 
    
    Arguments:
        batch_size: The size of the batch. Should be either a tensorflow tensor
                    and or an int.
        num_units:  Determines the number of units in the
                    lstm cell.
        activation: The activation function applied to the cells.
    Return:
        Returns the LSTM cell tensor object and the cell initialization matrix
    '''
    cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, state_is_tuple=True,
                                   activation=activation)
    cell_init = cell.zero_state(batch_size, dtype=tf.float32)
    
    return cell, cell_init

'''
The sequence_input is used to input the sequence as a matrix where
- The first dimension is used for batch
- The second dimension is used for time i.e. 5 (minutes, secs)
- The third dimension is used for the features of each token where a token is
  the complete features for a timestep
  
  Let's say packets are being fed into a server for 100 secs and each packet
  can be described by 2 features. If the batch size is 32 then this is fed
  into the model with shape [32, 100, 2].
'''
graph = tf.Graph()
with graph.as_default():
    SEQUENCE_INPUT = tf.placeholder(tf.float32, shape=(None, None, 2))
    LABEL_INPUT = tf.placeholder(tf.int32, shape=(None, 1))
    LABEL_ONE_HOT = tf.one_hot(LABEL_INPUT, L_UNIQUE)
    DROPOUT_IN = tf.placeholder_with_default(1.0, ())

    batch_size = tf.shape(SEQUENCE_INPUT)[0]

    CELL_FW, CELL_FW_INIT = create_lstm(batch_size, 32, activation=tf.nn.relu)
    CELL_BW, CELL_BW_INIT = create_lstm(batch_size, 32, activation=tf.nn.relu)

    x = SEQUENCE_INPUT

    outputs, states = bidirectional_dynamic_rnn(cell_fw=CELL_FW,
                                                cell_bw=CELL_BW,
                                                initial_state_fw=CELL_FW_INIT,
                                                initial_state_bw=CELL_BW_INIT,
                                                dtype=tf.float32,
                                                inputs=x)

    output_fw, output_bw = outputs
    outputs = (output_fw[:, -1, :], output_bw[:, -1, :])

    x = tf.concat(outputs, -1)
    x = tf.nn.dropout(x, DROPOUT_IN)
    x = tf.contrib.layers.fully_connected(x, 256)
    x = tf.nn.dropout(x, DROPOUT_IN)
    x = tf.contrib.layers.fully_connected(x, L_UNIQUE, activation_fn=None)
    PREDICTION_TENSOR = tf.contrib.layers.softmax(x)

    PREDICTED_LABEL_T = tf.round(PREDICTION_TENSOR)
    EQUAL_T = tf.equal(PREDICTED_LABEL_T, LABEL_ONE_HOT)
    ACCURACY_T = tf.reduce_mean(tf.cast(EQUAL_T, tf.float32))

    LOSS = tf.reduce_mean(tf.square(LABEL_ONE_HOT-PREDICTION_TENSOR))
    OPTIMIZER = tf.train.AdamOptimizer(1e-2).minimize(LOSS)

'''
Several constants are used to control the experiment.
BATCH_SIZE: The size of the batch used for training
EPOCHS: The number of epochs used to train
REPORT: The frequency of reporting the loss
SEQUENCE_LENGTH: The length of the sequence
'''

DATA_SIZE = len(sequence_windows)
BATCH_SIZE = 2**12
TEST_SIZE = 10
EPOCHS = 1
REPORT = DATA_SIZE//10
SEQUENCE_LENGTH = 10
TEST_SIZE = DATA_SIZE//10

TRAIN_FEATURES = sequence_windows
TRAIN_TARGETS = np.array(sequence_targets).reshape((-1, 1))

TEST_FEATURES = test_windows
TEST_TARGETS = np.array(test_targets).reshape((-1, 1))
TEST_LABELS = test_labels

with tf.Session(graph=graph) as session, trange(EPOCHS) as epoch_it:
    session.run(tf.global_variables_initializer())
    for epoch in epoch_it:
        for i, (low, high) in enumerate(range_batch(0, len(TRAIN_FEATURES), BATCH_SIZE)):
            
            X = TRAIN_FEATURES[low:high].reshape((-1, SEQUENCE_LENGTH, 2))
            Y = TRAIN_TARGETS[low:high]

            prev_L = session.run(LOSS, feed_dict={SEQUENCE_INPUT: X, LABEL_INPUT: Y})
            feed = {SEQUENCE_INPUT: X, LABEL_INPUT: Y, DROPOUT_IN: 0.8}
            session.run(OPTIMIZER, feed_dict=feed)
            new_L = session.run(LOSS, feed_dict={SEQUENCE_INPUT: X, LABEL_INPUT: Y})

        #epoch_it.write("Loss: {:.5f} -> {:.5f}".format(prev_L, new_L))
    
        accuracy_total = []
        true_positive = [0]*5
        false_positive = [0]*5
        scores = []
        for low, high in range_batch(0, len(TEST_FEATURES), BATCH_SIZE):

            X = TEST_FEATURES[low:high].reshape((-1, SEQUENCE_LENGTH, 2))
            Y = TEST_TARGETS[low:high]
            true_labels = TEST_LABELS[low:high]
            
            feed = {SEQUENCE_INPUT: X, LABEL_INPUT: Y}
            accuracy = session.run(ACCURACY_T, feed_dict=feed)
            accuracy_total.append(accuracy)
            
            feed = {SEQUENCE_INPUT: X}
            predictions = session.run(PREDICTION_TENSOR, feed_dict=feed)
            
            predict_length = len(predictions)
            inverse_predictions = 1 - predictions
            inverse_true_predictions = inverse_predictions[range(predict_length), Y.flatten()]
            inverse_true_predictions = inverse_true_predictions.reshape((-1, 1))
            
            scores.append(inverse_true_predictions)
            
        scores = np.concatenate(scores)
        auc = roc_auc_score(TEST_LABELS, scores)
        
        epoch_it.set_description("AUC: {:.6f}".format(auc))

from sklearn.metrics import roc_curve,auc
 
fpr, tpr, threshold = roc_curve(TEST_LABELS, scores)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()