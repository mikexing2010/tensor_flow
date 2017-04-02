
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from rnn_cell import RNNCell
from gru_cell import GRUCell

import argparse
import logging
import sys
import time
from datetime import datetime
import preprocessing
import cPickle
import os
import getopt
import tensorflow as tf
import numpy as np
from util import read_conll, one_hot, ConfusionMatrix, load_word_vector_mapping
from collections import *
from data_util import *
import time
'''
This model is a simple 2 layer feed forward network

y = softmax(z2)
z2 = RELU(W2x + b2)
z1 = RELU(W1x + b1)
'''

class Config:
    max_length = 600  # longest sequence to parse
    n_classes = 13
    dropout = 0.5
    embed_size = 100
    hidden_size =300
    batch_size = 400
    n_epochs = 30
    lr = 0.001

    def __init__(self):
        pass


class RNNModel():
    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape=[None, self.config.max_length], dtype=tf.int32)
        self.labels_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.lengths_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
        print self.pretrained_embeddings.shape

    def __init__(self, data, pretrained, config):
        self.documents, self.lengths, self.labels, _ = data[0]
        self.dev_examples, self.dev_lengths, self.dev_labels, _ = data[1]

        self.test_examples, self.test_lengths, self.test_labels, _ = data[2]
        self.pretrained_embeddings = pretrained
        self.config = config
        self.num_batch = int(len(self.documents)/self.config.batch_size)

        # Regularization
        self.regularizer = tf.contrib.layers.l2_regularizer(scale = 0.03)
        self.place = tf.placeholder(dtype=tf.float32, shape=self.pretrained_embeddings.shape)
        self.embedding_feed  = {}
        self.embedding_feed[self.place] = self.pretrained_embeddings
  
        # Build the model 
        self.add_placeholder()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

        self.print_train_loss_list = []
        self.print_dev_acc_list = []
        self.print_test_loss_list = []
        self.print_dev_loss_list = []

    def create_feed_dict(self, documents, label, lengths):
        feed_dict = {}
        if documents is not None:
            feed_dict[self.input_placeholder] = documents
        if label is not None:
            feed_dict[self.labels_placeholder] = label
        if lengths is not None:
            feed_dict[self.lengths_placeholder] = lengths
        if self.pretrained_embeddings.any:
            feed_dict[self.place] = self.pretrained_embeddings
        return feed_dict


    def add_embedding(self):
        #print "inside add_embedding"
        embeddings = tf.Variable(self.place)
        flattened = tf.reshape(self.input_placeholder, [-1])
        look_up = tf.nn.embedding_lookup(params=embeddings, ids=flattened)
        embeddings = tf.reshape(look_up, [-1, self.config.max_length, self.config.embed_size])
        #print(embeddings.get_shape())

        # Add embeddings and take average
        length = tf.expand_dims(tf.cast(self.lengths_placeholder, tf.float32), 1)
        embeddings = tf.div(tf.reduce_sum(embeddings, reduction_indices = 1), length)
        #print(embeddings.get_shape())
        # TODO: Need to average all the embeddings. Need length parameter to know
        # For each document in each batch, we need to look at the length
        # parameter to determine how many words in this document we need to
        # average  
        return embeddings

    def add_prediction_op(self):
        print "add_prediction_op"
        x = self.add_embedding()

        # TODO, use xavier as initializer? 
        xavier = tf.contrib.layers.xavier_initializer()
        W_1 = tf.get_variable("W_1", initializer = xavier, shape =[self.config.embed_size, self.config.hidden_size], dtype=tf.float32)
        b_1 = tf.get_variable("b_1", initializer = xavier, shape =[self.config.batch_size, self.config.hidden_size], dtype=tf.float32)
        W_2 = tf.get_variable("W_2", initializer = xavier, shape =[self.config.hidden_size, self.config.n_classes], dtype=tf.float32)
        b_2 = tf.get_variable("b_2", initializer = xavier, shape =[self.config.batch_size, self.config.n_classes], dtype = tf.float32)

        z = tf.nn.relu(tf.matmul(x, W_1) + b_1)
        z_dropout = tf.nn.dropout(z, keep_prob=self.config.dropout)
        pred = tf.nn.relu(tf.matmul(z_dropout, W_2) + b_2)
        return pred

    def add_loss_op(self, pred, regu_loss = 0):
        # TODO: is this loss ok with feedforward? 
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder,
logits=pred))
        return loss

    def add_training_op(self, loss):
        #train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def gen_batch(self, raw_x, raw_y, lengths):
        raw_x = np.array(raw_x)
        raw_y = np.array(raw_y)
        raw_lengths = [self.config.max_length if x > self.config.max_length else x for x in lengths]
        batch_size = self.config.batch_size
        # partition raw data into batches and stack them vertically in a data matrix
        # TODO: add 1 to batch_partition_length, otherwise we are ignoring data close to the end date --GeniusPanda
        data_length = len(raw_y)

        # ditching the last set
        if len(raw_x) % self.config.batch_size != 0:
            number_batches =  int(len(raw_x)/self.config.batch_size) - 1
        else:
            number_batches =  int(len(raw_x)/self.config.batch_size)
        data_x = []
        data_y = []
        lengths_docs = []
        for i in range(number_batches):
            data_x.append(raw_x[i*batch_size: (i+1)*batch_size])
            data_y.append(raw_y[i*batch_size: (i+1)*batch_size])
            lengths_docs.append(raw_lengths[i*batch_size : (i+1)*batch_size])
        return data_x, data_y, lengths_docs


    def print_helper(self, file):
        file.write("max_length: " + str(self.config.max_length) + "\n")
        file.write("embed_size:" + str(self.config.embed_size) + "\n")
        file.write("classes: " + str(self.config.n_classes) + "\n")
        file.write("hidden_size: " + str(self.config.hidden_size) + "\n")
        file.write("n_epochs: " + str(self.config.n_epochs) + "\n")
        file.write("learn_rate: "+ str(self.config.lr) + "\n")
        file.write("batch_size: " + str(self.config.batch_size)+ "\n")
        file.write("layers: " + str(1) + "\n")
        file.write("num_buckets: " + str(self.num_batch) + "\n")
        file.write("message for this run: " + str(message) + "\n")
        file.flush()

    def train(self, file_print, output_dir):
        self.print_helper(file_print)
        data_x, data_y, lengths_docs = self.gen_batch(self.documents, self.labels, self.lengths)
        #### data_x shape: num_batches x batch_size x max_length
        #### data_y shape: num_batches x batch_size x 1

        dev_x, dev_y, dev_lengths_docs = self.gen_batch(self.dev_examples, self.dev_labels, self.dev_lengths)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        weights_dir = output_dir + "/weights"
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)

        with tf.Session() as session:
            session.run(init, feed_dict=self.embedding_feed)
            for i in range(self.config.n_epochs):
                average_loss = 0
                print "-----epoch no. : " + str(i)
                for step, batch_data in enumerate(data_x):  # enumerating over batches
                    print "batch no. : " + str(step)
                    one_batch_inputs, one_batch_labels, lengths = batch_data, data_y[step], lengths_docs[step]
                    # print  one_batch_labels
                    feed = self.create_feed_dict(one_batch_inputs, one_batch_labels, lengths)
                    _, loss_val = session.run([self.train_op, self.loss], feed_dict=feed)
                    average_loss += loss_val
		    self.print_train_loss_list.append(loss_val)
                    # print np.mean(loss_val)
                average_loss /= self.num_batch
                print("Average loss at epoch", i, ": ", average_loss)
                    
                avg_loss, accuracy = self.evaluate_examples(saver, session, dev_x, dev_y, dev_lengths_docs, file_print)
          
                print "dev average prediction loss is ", average_loss
                self.print_dev_loss_list.append(average_loss)
                print "dev accuracy is ", accuracy
                self.print_dev_acc_list.append(accuracy)
        file_print.write("train_loss_per_batch: "+ str(self.print_train_loss_list)+"\n")
        file_print.write("dev_loss: "+ str(self.print_dev_loss_list)+ "\n")
        file_print.write("dev_acc: " + str(self.print_dev_acc_list) + "\n")
        file_print.close()

    def evaluate_examples(self, saver, sess, examples, labels, lengths, file):
        loss = 0
        correct_pred = 0
        all_pred = []
        all_labels = []
        step = 0
        print "evaluate examples"
        for step, batch_data in enumerate(examples):
            one_batch_inputs, one_batch_labels, leng = batch_data, labels[step], lengths[step]
            feed = self.create_feed_dict(one_batch_inputs, one_batch_labels, leng)
            pred, loss_val = sess.run([self.pred,  self.loss], feed_dict=feed)
            # pred is (batchsize * num_classes)
            # print "Prediction before argmax: ", pred
            pred = np.argmax(pred, axis=1)

            print "Prediction: ", pred
            all_pred.extend(pred)
            print "Labels: ", labels[step]
            all_labels.extend(labels[step])
            correct_pred += sum(pred == one_batch_labels)
            loss += loss_val
            if step % 100 == 1:
                print "predict ", step
        total_pred = len(self.dev_examples)
        accuracy = correct_pred / total_pred
        average_loss = loss/(step + 1)

        file.write("all preds: "+ "\t" + ", ".join(str(x) for x in all_pred)+ "\n")
        file.write("all lables: " + "\t" + ", ".join(str(x) for x in all_labels)+ "\n")

        return (average_loss, accuracy)


def load_word_vector_mapping(vocab_name, vector_name):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()

    with open(vocab_name, "rb") as reader:
        with open(vector_name, "rb") as reader2:
            for vocab, vector in zip(reader.readlines(), reader2.readlines()):
                vocab = vocab.strip()
                vector = vector.strip()
                ret[vocab] = list(map(float, vector.split()))
            return ret


def my_load_embeddings(our_dict):
    embeddings = np.array(np.random.randn(len(our_dict) + 1, Config.embed_size), dtype=np.float32)
    for word, vec in load_word_vector_mapping("./uranus/data/vocab.txt", "./uranus/data/wordVectors.txt").items():
        word = normalize(word)
        if word in our_dict:
            embeddings[our_dict[word]] = vec
    return embeddings


def load_word_vector_mapping_glove(vector_name):
    """
    Load word vector mapping using @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()
    with open(vector_name, "rb") as reader:
        for vector in reader.readlines():
            vector = vector.strip()
            list_vec = vector.split()
            ret[list_vec[0]] = map(float, list_vec[1:])

        return ret

def my_load_embeddings_glove(our_dict):
    print len(our_dict), "length of our dict"
    embeddings = np.array(np.random.randn(len(our_dict) + 1, Config.embed_size), dtype=np.float32)
    for word, vec in load_word_vector_mapping_glove("../glove.6B.100d.txt").items():
        word = normalize(word)
        if word in our_dict:
            embeddings[our_dict[word]] = vec
    return embeddings

    '''
        only contains lower_case and NUM for numbers
        tok2id = {}
        word: ranks of frequency, starting from 1 (1 is the most freq)

    '''

def do_train():
    try:
        if process_large == True:
            train_path = open("./big_pickled_files_100D/train", "rb")
            dev_path = open("./big_pickled_files_100D/dev", "rb")
            test_path = open("./big_pickled_files_100D/test", "rb")
        elif process_small == True:
            train_path = open("./small_pickled_files/train", "rb")
            dev_path = open("./small_pickled_files/dev", "rb")
            test_path = open("./small_pickled_files/test", "rb")
        elif process_medium == True:
            train_path = open("./medium_pickled_files_100D/train", "rb")
            dev_path = open("./medium_pickled_files_100D/dev", "rb")
            test_path = open("./medium_pickled_files_100D/test", "rb")
    except IOError:
        print "Could not open file!" 
        sys.exit()    
    dev = cPickle.load(dev_path)
    test = cPickle.load(test_path)
    train = cPickle.load(train_path)

    # TODO: change this to glove eventually
    embeddings = my_load_embeddings_glove (train[3])
    print_files = "./print_files"
    if not os.path.isdir(print_files):
        os.makedirs(print_files)

    if process_small == True:
        output_dir = print_files + "/{:%Y%m%d_%H%M%S}_small".format(datetime.now())
    elif process_medium == True:
        output_dir = print_files + "/{:%Y%m%d_%H%M%S}_medium".format(datetime.now())
    elif process_large == True:
        output_dir = print_files + "/{:%Y%m%d_%H%M%S}_large".format(datetime.now())

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    file_print = open(output_dir + "/run_result.txt", "wrb")
    all_data = (train, dev, test)
    # This takes a long time because it's building the graph
    start = time.time()
    our_model = RNNModel(data=all_data, pretrained=embeddings, config=Config)
    elapsed = (time.time() - start)
    print "BUILDING THE MODEL TOOK " + str(elapsed) + "SECS"
    our_model.train(file_print, output_dir)


def main(argv):
    global process_small
    global process_medium
    global process_large
    global message
    process_small = False
    process_medium = False
    process_large = False
    message = ""
    try:
        opts, args = getopt.getopt(argv,"hsmlx:",["help","small", "medium","large", "message"])
    except getopt.GetoptError:
        print 'test.py [-h|help] [-s|small] [-m|medium] [-l|large] -x'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print 'test.py [-h|--help] [-s|--small] [-m|--medium] [-l|--large] -x'
            sys.exit()
        elif opt in ("-s", "--small"):
            process_small = True
        elif opt in ("-m", "--medium"):
            if process_small == True:
                print "error: can chooose only one data size"
                sys.exit()
            process_medium = True
        elif opt in ("-l", "--large"):
            if process_small == True or process_medium == True:
                print "error: can chooose only one data size"
            process_large = True
        elif opt in ("-x", "--message"):
            message = arg

    if process_small == False and process_medium == False and process_large == False:
        print "must specify the size of the files to process: -s -m or -l"
        sys.exit()
    if message == "":
        print "Error: must use -m option to explain in a message what you are running"
        sys.exit()

    do_train()

if __name__ == "__main__":
   main(sys.argv[1:])
