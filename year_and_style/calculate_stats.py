import sampling
import numpy as np
import matplotlib.pyplot as plt
import math
import nltk

def length(directory_path):
    '''
    This functions counts all the words in every doc and
        shows a histogram that represents a the freq of these
        counts.

    :param directory_path: path to directory with original text files
    :return: Shows a histogram of the word counts of the docs.
    '''

    '''
    with punctuation:
    -600, -1200, - 1700...

    Most are -600.



    '''


    pairs = sampling.list_dates(directory_path)
    years, paths = zip(*pairs)
    length_list = []
    for path in paths:
        with open(path, "r") as reader:
            feed = reader.read().decode('utf-8')
            num_lines = len(nltk.tokenize.word_tokenize(feed))
            length_list.append(num_lines)
    ## put into buckets and make histogram ##
    bins = np.linspace(math.ceil(min(length_list)),
                   math.floor(max(length_list)), 100)
    plt.xlim([min(length_list)-5, max(length_list)+5])

    plt.hist(length_list, bins=bins, alpha=0.5)
    plt.title('Distribution of the Doc Length Frequencies')
    plt.xlabel('variable X')
    plt.ylabel('count')
    plt.savefig("./distribution_doc_length")
    plt.show()

if __name__ == "__main__":
    directory_path = "/Users/eunseo/Desktop/frus/docs_by_date_raw_parse/"
    length(directory_path)