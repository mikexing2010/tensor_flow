import nltk
import sys
import getopt
import os
import re
import tensorflow as tf
from data_util import *
import random
import cPickle
import argparse
import getopt
from nltk.corpus import stopwords


'''
 13 classes total
'''


class Config:

    start_date = 1861
    end_date = 1983
    bucket_size = 10
    num_buckets = (end_date-start_date)/bucket_size + 1
    max_length = 600

def process(old_dir, new_dir):
    '''
    this function takes in paths to the old and new directories, processes the old files
        and writes them in the new directory

    :param old_dir: path to the old_directory that has all the original files
    :param new_dir: path to the new_directory that will output all the processed files

    :return: None
    '''

    files = os.listdir(old_dir)
    paths = map(lambda x: os.path.join(old_dir, x), files)
    new_paths = map(lambda x: os.path.join(new_dir, x), files)

    for f in zip(paths, new_paths):
        with open(f[0], "r") as reader:
            tokenized = nltk.tokenize.word_tokenize(reader.read().decode("utf-8"))
            tokenized = [normalize(word) for word in tokenized]
            #with open(f[1], "w") as writer:
            #    writer.write(" ".join(tokenized).encode("utf-8"))

def make_vocab(dir):
    '''
    This function creates a vocab.text file that contains all the unique vocab of given text

    :param dir: path to directory where the files can be found
    :return: None
    '''
    files = os.listdir(dir)
    paths = map(lambda x: os.path.join(dir, x), files)
    output = os.path.join("./pluto_data", "vocab.txt")

    all_vocab = set()

    for f in paths:
        with open(f, "r") as file_read:
            words = map(lambda x: x.decode("utf-8"), file_read.read().split())
            all_vocab |= set(words)

    with open(output, "w") as file_write:
        for v in all_vocab:
            file_write.write(v.encode("utf-8")+"\n")


def vocab_dict(path_to_vocab):
    '''
    :param path_to_vocab: path to the vocab.txt file
    :return: dictionary that maps the key word to the value assigned to it (in order of listing on vocab.txt -1 )

    '''
    dict = {}
    with open(path_to_vocab, "r") as reader:
        for r in enumerate(reader):
            dict[r[1][:-1]] = r[0]
    total_v = len(dict.keys())
    return dict, total_v

def vocab_dict_glove(path_to_glove):
    '''
    :param path_to_vocab: path to the glove file
    :return: dictionary that maps the key word to the value assigned to it (in order of listing on vocab.txt -1 )

    '''
    dict = {}
    with open(path_to_glove, "r") as reader:
        for r in enumerate(reader):
            word = r[1].strip().split()[0]
            dict[word] = r[0]
    total_v = len(dict.keys())
    return dict, total_v


def rewrite_in_ints(dict, old_dir, total_vocab):
    '''
    This function takes in the vocab-int mapping and rewrites the original text to reflect this.
    It also returns a list of tuples of x and y.

    :param dict: dictionary mapping vocab to int
    :param dir: path to the directory
    :param numerical_dir: path to the new directory to print ints
    :param total_vocab: num of vocab (to pass to one-hot function)
    :return: x and y data, and lengths of the prepadded docs

    '''
    	
    files = os.listdir(old_dir)
    paths = map(lambda x: os.path.join(old_dir, x), files)
    stopW = set(stopwords.words('english'))
    punc = ".,;:?!"

    
    count_randoms = 0
    #new_paths = map(lambda x: os.path.join(numerical_dir, x), files)

    all_years = []
    all_x = []

    for f in paths:
        with open(f, "r") as reader:
            tokenized = nltk.tokenize.word_tokenize(reader.read().decode("utf-8"))
            tokenized = [normalize(word) for word in tokenized]
	    if len(tokenized) ==0:
		continue
            ints = []
            for x in tokenized:
                if x in dict and (x in stopW or x in punc):
                    ints.append(dict[x])
                else:
		    count_randoms += 1
                    #random_id = np.random.randint(0, high=len(dict.keys()))
                    zsomber = 399998
		    ints.append(zsomber)
            ## taking the year of the file
            year_grab = re.compile("(\d\d\d\d).+")
            print f, "print file name"
	    year = year_grab.search(f).group(1)
            #matrix = convertToOneHot(np.array(map(int, ints)), total_vocab)
            all_years.append(year)
            all_x.append(ints)
            ###to write file
            #with open(f[1], "w") as writer:
            #    writer.write(" ".join(ints))
    int_years = year_buckets(all_years)

    #years = tf.one_hot(int_years, Config.num_buckets, on_value=1.0, off_value=0.0, axis=-1)
    ### years returns a matrix of Num_DataPoints x Num_buckets
    truncated_x, lengths = truncate_or_pad(all_x)
    ### matrices returns a matrix of Num_DataPoints x vocab_size*maxlength
    print "number of missing words, randomized: ", str(count_randoms)
    return truncated_x, lengths, int_years

def year_buckets(list_years):
    return [(int(i)-Config.start_date)/Config.bucket_size for i in list_years]

#!/usr/bin/env python
import numpy as np

# def convertToOneHot(vector, num_classes=None):
#     """
#     Converts an input 1-D vector of integers into an output
#     2-D array of one-hot vectors, where an i'th input value
#     of j will set a '1' in the i'th row, j'th column of the
#     output array.
#
#     Example:
#         v = np.array((1, 0, 4))
#         one_hot_v = convertToOneHot(v)
#         print one_hot_v
#
#         [[0 1 0 0 0]
#          [1 0 0 0 0]
#          [0 0 0 0 1]]
#
#
#
#     >>> a = np.array([1, 0, 3])
#
#     >>> convertToOneHot(a)
#     array([[0, 1, 0, 0],
#            [1, 0, 0, 0],
#            [0, 0, 0, 1]])
#
#     >>> convertToOneHot(a, num_classes=10)
#     array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
#     """
#
#     assert isinstance(vector, np.ndarray)
#     assert len(vector) > 0
#
#     if num_classes is None:
#         num_classes = np.max(vector)+1
#     else:
#         assert num_classes > 0
#         assert num_classes >= np.max(vector)
#
#     result = np.zeros(shape=(len(vector), num_classes))
#     result[np.arange(len(vector)), vector] = 1
#     return result.astype(int)

def truncate_or_pad(data):
    '''

    :param data: list of tuples with the second element being list of one-hot vectors
    :param max_length: the cut-off for rnn- Time steps
    :return: truncated list, list of lengths
    '''
    lengths = []
    '''
    data = np.array(data)
    for i in range(len(data)):
        if len(data[i]) > Config.max_length:
            lengths.append(data[i].shape[0])
            data[i] = tf.constant(data[i][:Config.max_length])
        elif len(data[i]) < Config.max_length:
            lengths.append(data[i].shape[0])
            data[i] = tf.pad(data[i], np.array([[0,Config.max_length-data[i].shape[0]],[0,0]]), "CONSTANT")
    return data, lengths
    '''

    for doc in range(len(data)):
        if len(data[doc]) > Config.max_length:
            lengths.append(len(data[doc]))
            data[doc] = data[doc][:Config.max_length]
        elif len(data[doc]) < Config.max_length:
            lengths.append(len(data[doc]))
            data[doc] = [data[doc][i] if i < len(data[doc]) else 0.0 for i in range(Config.max_length)]
    return data, lengths

def make_dirs(root):
    '''
    :param root: path to the root
    :return paths to processed and numerical doc directories (with train, dev, and test directories inside)
    '''
    processed = root + "_processed"
    numerical = root + "_numerical"
    if not os.path.isdir(processed):
        os.makedirs(processed)
        os.makedirs(numerical)

    for set_type in ["test","dev","train"]:
        if not os.path.isdir(os.path.join(processed, set_type)):
            os.makedirs(os.path.join(processed, set_type))
            os.makedirs(os.path.join(numerical, set_type))

    return processed, numerical

def pretrained_embeddings(root_path, pretrained_vocab_path, pickle_path):
    '''

    :param    root_path: path to the root directory
    :param    pretrained_vocab_path: path to unique vocab path

    :return   None

            pickles the following data to pickle path per set-type(test,dev,train)
              truncated_X:  all words of docs as ints
              lengths:      vectors of lengths of docs before padding
              years:        the y labels for all docs (order corresponds to the trauncated_x)
              dict:         vocab-key to int-value dictionary
    '''
    dict, total_vocab = vocab_dict(pretrained_vocab_path)
    processed, numerical = make_dirs(root_path)
    for set_type in ["test","dev","train"]:
        process(os.path.join(root_path, set_type), os.path.join(processed, set_type))
        truncated_x, lengths, years = rewrite_in_ints(dict, os.path.join(processed, set_type),
                                       os.path.join(numerical, set_type), total_vocab)

        f = open(os.path.join(pickle_path, set_type), "wb")
        cPickle.dump((truncated_x, lengths, years, dict), f)
        # truncated_x has dimensions:
        #    num_docs x Config.max_length x Vocab
    print "Finished preprocessing!"



def pretrained_embeddings_glove(root_path, pretrained_vocab_path, pickle_path):
    '''

    :param    root_path: path to the root directory
    :param    pretrained_vocab_path: path to unique vocab path

    :return   None

            pickles the following data to pickle path per set-type(test,dev,train)
              truncated_X:  all words of docs as ints
              lengths:      vectors of lengths of docs before padding
              years:        the y labels for all docs (order corresponds to the trauncated_x)
              dict:         vocab-key to int-value dictionary
    '''
    dict, total_vocab = vocab_dict_glove(pretrained_vocab_path)
    for set_type in ["test","dev","train"]:
        truncated_x, lengths, years = rewrite_in_ints(dict, os.path.join(root_path, set_type), total_vocab)
        f = open(os.path.join(pickle_path, set_type), "wb")
        cPickle.dump((truncated_x, lengths, years, dict), f)
        # truncated_x has dimensions:
        #    num_docs x Config.max_length x Vocab
    print "Finished preprocessing!"


def not_pretrained():

    old = "./working_set/train"
    new = "./working_set_processed/train"
    numeri = "./working_set_numerical/train"
    process(old, new)
    make_vocab(new)
    dict, total_vocab = vocab_dict("./pluto_data/vocab.txt")
    truncated_x, lengths, years = rewrite_in_ints(dict, new, numeri, total_vocab)
    print "Finished preprocessing!"
    return truncated_x, lengths, years, dict


def main(argv):
    local_root_path = ""
    pickled_root_path = "stop_words_rnn_pickles"
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["help","input", "output"])
    except getopt.GetoptError:
        print 'test.py [-h|help] [-i|input_dir] [-o|pickled_output_dir]'
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print 'preprocessing.py [-h|--help] [-i|-input_dir] [-o|--pickled_output_dir]'
            sys.exit()
        elif opt in ("-i", "--input_dir"):
            local_root_path = arg
        elif opt in ("-o", "--output_dir"):
            pickled_root_path = arg

    if local_root_path == "" or pickled_root_path == "":
        print 'preprocessing.py [-h|--help] [-i|-input_dir] [-o|--pickled_output_dir] check'
        sys.exit(2)

    gpu_root_path = "/home/teamjo/stylometry/main_data"
    pretrained_vocab_path = "/home/teamjo/stylometry/glove.6B.100d.txt"
    if not os.path.isdir(gpu_root_path):
        print "Error: input dir ", gpu_root_path, " does not exist"
        sys.exit(2)

    if not os.path.isdir(pickled_root_path):
        os.makedirs(pickled_root_path)
    pretrained_embeddings_glove(gpu_root_path, pretrained_vocab_path, pickled_root_path)

if __name__ == "__main__":
   main(sys.argv[1:])
