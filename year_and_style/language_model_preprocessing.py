from __future__ import division, unicode_literals
import re
import cPickle
import os
import nltk
import math

from textblob import TextBlob as tb


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)




'''
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
'''

global year_counts
year_counts = {}

class Config:
     
    year_range = [1940,1949]
    start_date = 1861
    end_date = 1983
    bucket_size = 10
    num_buckets = (end_date-start_date)/bucket_size + 1
    max_length = 600
    cap_docs = 5000


def read_files(path):
    all_files = os.listdir(path)
    path_files = [os.path.join(path, i) for i in all_files]
    print path_files


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

def expand_range(year_range):
    years = set()
    start = year_range[0]
    for i in range(year_range[1]-year_range[0]+1):
        years.add(start+i)
    return years

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

    files = sorted(os.listdir(old_dir))
    #files = os.listdir(old_dir)
    paths = map(lambda x: os.path.join(old_dir, x), files)
    count_randoms = 0
    #new_paths = map(lambda x: os.path.join(numerical_dir, x), files)
    year_range = expand_range(Config.year_range) 
    all_years = []
    all_x = []
    total_tokens = 0
    for f in paths:
        with open(f, "r") as reader:
            print f
            print "hi"
            tokenized = nltk.tokenize.word_tokenize(reader.read().decode("utf-8"))
            tokenized = [normalize(word) for word in tokenized]
            tok = len(tokenized)
            total_tokens += tok
            if tok <= 0:
                continue
            year_grab = re.compile("(\d\d\d\d).+")
            print f, "print file name"
            year = year_grab.search(f).group(1)
            if int(year) not in year_range:
	        continue
	    global year_counts
            if year not in year_counts:
                year_counts[year] = 1
            elif year in year_counts:
                co = year_counts[year]
                if co == 5000:
                    continue
                else:
                    co += 1
                    year_counts[year] = co

            ints = []
            for x in tokenized:
                if x in dict:
                    ints.append(dict[x])
                else:
                    count_randoms += 1
                    #random_id = np.random.randint(0, high=len(dict.keys()))
                    zsomber = 99997
                    ints.append(zsomber)
            ## taking the year of the file

            #matrix = convertToOneHot(np.array(map(int, ints)), total_vocab)
            all_years.append(int(year))
            all_x.append(ints)
            ###to write file
            #with open(f[1], "w") as writer:
            #    writer.write(" ".join(ints))

    #years = tf.one_hot(int_years, Config.num_buckets, on_value=1.0, off_value=0.0, axis=-1)
    ### years returns a matrix of Num_DataPoints x Num_buckets
    truncated_x, lengths = truncate_or_pad(all_x)
    ### matrices returns a matrix of Num_DataPoints x vocab_size*maxlength
    print "number of missing words, randomized: ", str(count_randoms)
    return truncated_x, lengths, all_years


def run_preprocessing(pretrained_embeddings_path, path_to_text, pickle_path):
    dict, total_vocab = vocab_dict_glove(pretrained_embeddings_path)
    truncated_x, lengths, years = rewrite_in_ints(dict, path_to_text, total_vocab)
    f = open(pickle_path, "wb")
    cPickle.dump((truncated_x, lengths, years, dict), f)
    # truncated_x has dimensions:
    #    num_docs x Config.max_length x Vocab
    print "Finished preprocessing!"

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

def normalize(word):
    """
    Noirmalize words that are numbers or have casing.
    """
    if word.isdigit(): return "nnumm"
    else: return word.lower()

if __name__ == "__main__":

    pickle_path = "./language_model_text/1940_600_smallest_glove_sorted"
    text_path = "../docs_by_date_raw_parse"
    embeddings_path = "../smallest_glove_100d.txt"
    run_preprocessing(embeddings_path, text_path, pickle_path)
