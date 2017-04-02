
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
import pickle
import os
import getopt
import tensorflow as tf
import numpy as np
from util import read_conll, one_hot, ConfusionMatrix, load_word_vector_mapping
from collections import *
from data_util import *
from gru_cell import GRUCell
import time
import copy
import cProfile, pstats

'''
TODO LIST:

1. GloVe Vectors
2. Print out Pred, Gold in file for Confusion Matrix
3. Confusion Matrix
3.5. Make the probability output the expectation, not the absolute.
4. Bag of Words Baseline
5. Re-do sampling to fit language model implementation
6. Start Language Model Implementation


Details about the Language Model Implementation:
Train for every decade, the first part of the
decade gets trained and then we test on the second part of the decade and this is done over time;
We expect to see perplexity increase at moment of new topical change.





"Hyperparameters:\\
Learning Rate
Dropout Probability
Optimization + Loss Functions
Hidden Layer Dimension\\
Embedding Dimensions (if not pretrained)\\
Dynamic_RNN\\
BucketSize\\
Document Length\\

GRU, LSTM, etc.


SUNDAY: EPochs, Dev, try to run on the GPU, test Results
Sanity Test everything

How many failed Look ups


Huge TODO Loss divide by ten?

'''


logger = logging.getLogger("baseline")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    max_length =600  # longest sequence to parse
    n_classes = 99999
    dropout = 0.5
    model_option = "dynamic"   ##  "dynamic" or "vanilla"
    embed_size = 100
    hidden_size = 200
    batch_size = 2
    n_epochs = 1
    #max_grad_norm = 10.
    lr = 0.001
    cell = "gru"   ## "gru" or "rnn"
    clip_gradients = True
    max_grad_norm = 5
    end_token = 99998
    pre1900 = 11502 
   
    def __init__(self):
        pass


class RNNModel():
    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape=[None, self.config.max_length], dtype=tf.int32)
        self.labels_placeholder = tf.placeholder(shape=[None, self.config.max_length], dtype=tf.int32)
        self.years_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.lengths_placeholder = tf.placeholder(shape=[None], dtype=tf.int32)
        #self.mask_placeholder = tf.placeholder(shape=[None, self.max_length], dtype=tf.bool)
        #self.dropout_placeholder = tf.placeholder(shape=[], dtype=tf.float32)

    def __init__(self, data, pretrained, config):
        print len(data), "length of data"
        self.documents = data[0]
        self.lengths = data[1]
        self.years = data[2]

        self.pretrained_embeddings = pretrained
        self.config = config
        self.num_batch = int(len(self.documents)/self.config.batch_size)

        #regularization
        self.regularizer = tf.contrib.layers.l2_regularizer(scale = 0.03)
        #self.l1 = tf.contrib.layers.l1_regularizer(scale = 0.03)

	self.place = tf.placeholder(dtype=tf.float32, shape=self.pretrained_embeddings.shape)
        self.add_placeholder()
        self.pred, self.regu_loss = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred, self.regu_loss)
        self.train_op = self.add_training_op(self.loss)
	
	self.embedding_feed = {}
	self.embedding_feed[self.place] = self.pretrained_embeddings
        self.print_running_average_list = []
        self.print_years = []
        self.print_train_acc_list = []
        self.print_test_loss_list = []
        self.print_train_loss_list = []

    def create_feed_dict(self, documents, label, lengths, years):
        feed_dict = {}
        if documents is not None:
            feed_dict[self.input_placeholder] = documents
        if label is not None:
            feed_dict[self.labels_placeholder] = label
        if self.config.model_option == "dynamic":
            feed_dict[self.lengths_placeholder] = lengths
        if years:
            feed_dict[self.years_placeholder] = years
        if self.pretrained_embeddings.any:
	    feed_dict[self.place] = self.pretrained_embeddings
	return feed_dict

    def add_embedding(self):
        embeddings = tf.Variable(self.place, "embeddings")
        flattened = tf.reshape(self.input_placeholder, [-1])
        look_up = tf.nn.embedding_lookup(params=embeddings, ids=flattened)
        # TODO: checking embedding done correctly? - Mewtwo
        embeddings = tf.reshape(look_up, [-1, self.config.max_length, self.config.embed_size])
        #### batch x 600 x 50

        return embeddings

    def add_prediction_op(self):

        x = self.add_embedding()

        if self.config.cell == "rnn":
            cell = RNNCell(self.config.embed_size, self.config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(self.config.embed_size, self.config.hidden_size, self.regularizer)

        if self.config.dropout != 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.config.dropout)
        lengths = self.lengths_placeholder

        xavier = tf.contrib.layers.xavier_initializer()
        U = tf.get_variable("U", initializer=xavier, shape=[self.config.hidden_size, self.config.n_classes],
                            dtype=tf.float32)
        b_2 = tf.get_variable("b_2", initializer=tf.zeros(shape=[1, self.config.n_classes], dtype=tf.float32))

        #if self.config.model_option == "vanilla":
            #z = self.vanilla_rnn(cell, x, U, b_2)
        if self.config.model_option == "dynamic":
            z, regu_loss = self.dynamic(cell, x, U, b_2, lengths)
        #preds = tf.transpose(tf.pack(preds), perm=[1, 0, 2])
        return z, regu_loss   ##z is batch x words x vocab

    def dynamic(self, cell, x, U, b_2, lengths):
        #x = tf.transpose(x, [1, 0, 2])

        # Reshaping to (n_steps*batch_size, n_input)
        #x = tf.reshape(x, [-1, self.config.embed_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.split(0, self.config.max_length, x)
        # Define a lstm cell with tensorflow
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(Config.hidden_size)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        #print lengths, "print lengths", "type: ",type(lengths)
        outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                    sequence_length=lengths)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input], n_input is hidden_layer_size
        #outputs = tf.pack(outputs)
        #outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        #batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        #index = tf.range(0, batch_size) * self.config.max_length + (lengths - 1)
        #a = tf.reshape(outputs,[-1,self.config.hidden_size])
	#outputs = tf.gather(a, index)
	# Indexing
        #outputs = tf.gather(tf.reshape(outputs, [-1, self.config.hidden_size]), index)
        print "output shape is ", outputs.get_shape()
        U = tf.expand_dims(U, axis=0)
        #print tf.shape(U)
        U = tf.tile(U, [self.config.batch_size, 1,1])
        print "U shape is ", U.get_shape()
        d = tf.batch_matmul(outputs, U)
        print "d shape is ", d.get_shape()
        c = d + b_2
        return c, 0#  tf.contrib.layers.apply_regularization(self.regularizer, [U]) +cell.regularization_loss

    def vanilla_rnn(self,cell, x, U, b_2):
        with tf.variable_scope("RNN"):
            h = tf.zeros(shape=[1, self.config.hidden_size], dtype=tf.float32)
            for time_step in range(self.config.max_length):
                ### YOUR CODE HERE (~6-10 lines)
                # Note that the output and h are the same for RNN.
                # Will be different for LSTM

                # taking the timestamp'th word of each sentence in a batch
                # which means take each column

                #inputs needs to be 5 x 50
                output, h = cell(x[:, time_step], h, scope="RNN")
                tf.get_variable_scope().reuse_variables()
                #h_drop = tf.nn.dropout(h, keep_prob=dropout_rate)
                # print z.get_shape()
                # add prediction that's before the softmax layer
                ### END YOUR CODE
            z = tf.matmul(h, U) + b_2
        return z

    def add_loss_op(self, pred, regu_loss = 0):
        #pred : batch x 600 x vocab
        total_loss = []
        for i in range(self.config.batch_size):
            print "add_loss_op ", i
            print self.labels_placeholder.get_shape()
            print self.lengths_placeholder[i]
            la = self.labels_placeholder[i][: self.lengths_placeholder[i]]
            pr = pred[i][: self.lengths_placeholder[i]]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=la, logits=pr)
            loss_for_one_doc = tf.reduce_mean(losses)
            total_loss.append(loss_for_one_doc)
        total_loss = tf.reduce_mean(total_loss)
        return total_loss ###vector of batchsize (each elemnt is the mean loss)

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        gradients = optimizer.compute_gradients(loss)
        gradients_only = [grad[0] for grad in gradients]
        variables = [grad[1] for grad in gradients]

        if self.config.clip_gradients:
            gradients_only, _ = tf.clip_by_global_norm(gradients_only, self.config.max_grad_norm)
            # combine variables and gradients together
            new_gradients = []
            for i in range(len(gradients)):
                new_gradients.append((gradients_only[i], variables[i]))
            gradients = new_gradients

        self.grad_norm = tf.global_norm(gradients_only)
        train_op = optimizer.apply_gradients(gradients)

        return train_op


    def gen_batch(self, raw_x, lengths, years):
        raw_x = np.array(raw_x)

        raw_lengths = [self.config.max_length if x > self.config.max_length else x for x in lengths]
        batch_size = self.config.batch_size
        # partition raw data into batches and stack them vertically in a data matrix
        # TODO: add 1 to batch_partition_length, otherwise we are ignoring data close to the end date --GeniusPanda

        if len(raw_x) % self.config.batch_size != 0:
            number_batches =  int(len(raw_x)/self.config.batch_size) #+ 1
        else:
            number_batches =  int(len(raw_x)/self.config.batch_size)
        data_x = []
        lengths_docs = []
        years_docs = []
        for i in range((number_batches)):
	    data_x.append(raw_x[i*batch_size: (i+1)*batch_size])
            lengths_docs.append(raw_lengths[i*batch_size : (i+1)*batch_size])
            years_docs.append(years[i*batch_size : (i+1)*batch_size])
        return data_x, lengths_docs, years_docs

    def get_dev(self):
        pass

    def get_test(self):
        pass


    def print_helper(self, file):
        file.write("max_length: " + str(self.config.max_length) + "\n")
        file.write("embed_size: " + str(self.config.embed_size) + "\n")
        file.write("classes: " + str(self.config.n_classes) + "\n")
        file.write("hidden_size: " + str(self.config.hidden_size) + "\n")
        file.write("n_epochs: " + str(self.config.n_epochs) + "\n")
        file.write("learn_rate: "+ str(self.config.lr) + "\n")
        file.write("batch_size: " + str(self.config.batch_size)+ "\n")
        file.write("layers: " + str(1) + "\n")
        file.write("num_buckets: " + str(self.num_batch) + "\n")
        file.write("cell type: " + str(self.config.cell) + "\n")
        file.write("clip_gradients: " + str(self.config.clip_gradients) + "\n")
        file.write("message for this run: " + str(message) + "\n")
        file.flush()

    def train(self, file_print, output_dir):
        self.print_helper(file_print)
        data_x, lengths_docs, years_docs = self.gen_batch(self.documents, self.lengths, self.years)
        #### data_x shape: num_batches x batch_size x max_length
        #### data_y shape: num_batches x batch_size x 1

        #dev_x, dev_y, dev_lengths_docs = self.gen_batch(self.dev_examples, self.dev_labels, self.dev_lengths)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        weights_dir = output_dir + "/weights"
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)

        with tf.Session() as session:
            session.run(init, feed_dict=self.embedding_feed)
            # data_x = [i.eval() for i in data_x]
            # data_y = [i.eval() for i in data_y]
            for i in range(self.config.n_epochs):
                print "epoch no. : " + str(i)
                for step, batch_data in enumerate(data_x):  # enumerating over batches
                    one_batch_labels = copy.deepcopy(batch_data)
                    one_batch_labels = np.array([np.hstack((i[1:],np.array([self.config.end_token])))for i in one_batch_labels])
                    print "batch no. : " + str(step)
		    one_batch_inputs, lengths, years = batch_data, lengths_docs[step], years_docs[step]
                    feed = self.create_feed_dict(one_batch_inputs, one_batch_labels, lengths, years)
                    _, loss_val, pred = session.run([self.train_op, self.loss, self.pred], feed_dict=feed)
                    print "session ran"
                    pred = np.argmax(pred, axis=2)
                    print "pred finished", pred
                    sum_corr = 0
                    sum_lengths = sum(lengths)
                    for k in range(len(one_batch_inputs)):
                        sum_corr += sum(pred[k][:lengths[k]] == one_batch_labels[k][:lengths[k]])
                    mean_average = loss_val
                    print "mean average", mean_average
                    accuracy_per_batch = float(sum_corr)/sum_lengths
                #avg_loss, accuracy = self.evaluate_examples(saver, session, dev_x, dev_y, dev_lengths_docs, file_print)
                    print "current year: ", years[0]
                    self.print_years.append(years[0])
                    print "batch average prediction loss is ", mean_average
                    self.print_train_loss_list.append(mean_average)
                    print "batch accuracy is ", accuracy_per_batch
                    self.print_train_acc_list.append(accuracy_per_batch)
	     	    if step%10 == 0: 
             		saver.save(session, os.path.join(weights_dir, str(step)))
			print "weights saved!"
            	    file_print.write("average_train_loss: "+ str(self.print_train_loss_list)+ "\n")
                    file_print.write("average_train_acc: " + str(self.print_train_acc_list) + "\n")
                    file_print.write("starting year per batch: " + str(self.print_years) + "\n")
                    file_print.flush()
	    file_print.close()
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
    embeddings = np.array(np.random.randn(len(our_dict) + 1, Config.embed_size), dtype=np.float32)
    for word, vec in load_word_vector_mapping_glove("./smallest_glove_100d.txt").items():
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
            train_path = open("./language_model_text/1940_600_smallest_glove_sorted", "rb")
        elif process_small == True:
            # TODO: the logic need to be changed to be more flexible
            train_path = open("./language_model_text/1940_600_smallest_glove_sorted", "rb")
        elif process_medium == True:
            train_path = open("./language_model_text/1940_600_smallest_glove_sorted", "rb")
    except IOError:
        print "Could not open file!"
        sys.exit()
    train = pickle.load(train_path)

    embeddings = my_load_embeddings_glove(train[3])
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

    start = time.time()
    elapsed = (time.time() - start)
    print "TEST that time works " + str(elapsed) + "SECS"
    file_print = open(output_dir + "/run_result.txt", "wrb")
    # This takes a long time because it's building the graph
    start = time.time()
    our_model = RNNModel(data=train, pretrained=embeddings, config=Config)
    elapsed = (time.time() - start)
    print "BUILDING THE MODEL TOOK " + str(elapsed) + "SECS"
    # TODO; model needs to be changed to Config.model_option
    # print "Built RNN Model with option: " + model  +" \n Built with the cell option: "+ Config.cell
    # file_print.write("Built RNN Model with option: " + model  +" \n Built with the cell option: "+ Config.cell)
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
    pr = cProfile.Profile()
    pr.enable()

    main(sys.argv[1:])
    pr.disable()
    ps = pstats.Stats(pr).sort_stats("cumtime")
    ps.print_stats()
