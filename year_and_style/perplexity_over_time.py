import os
import nltk
import re
import cPickle

##generate docs


global year_counts
year_counts = {}

class Config:

    text_path = "/Users/eunseo/Desktop/frus/docs_by_date_raw_parse"
    embeddings_path = "./smallest_glove_100d.txt"  #### has 99999 vocab
    pickle_dir = "../pickles_by_decade"

    start_year = 1890
    end_year = 1980
    buckets = 5
    per_year_cap = 5000
    not_a_word = 99998
    doc_lengths = 50
    train_count = 200000
    dev_count = 30000
    percentage_dev = .15

def make_year_buckets():
    year_buckets = {}
    bucket_count = 0
    for i in range(Config.end_year-Config.start_year):
        year_buckets[Config.start_year+i] = bucket_count
        if i % 5 == 4:
            bucket_count += 1
    return year_buckets

def makedir():
    if not os.path.isdir(Config.pickle_dir):
        os.makedirs(Config.pickle_dir)

def return_paths():
    files = sorted(os.listdir(Config.text_path), key=lambda x: get_year(x))
    paths = map(lambda x: os.path.join(Config.text_path, x), files)
    return paths, files

def embeddings_dict():
    vector_dict = {}
    with open(Config.embeddings_path, "r") as reader:
        for r in enumerate(reader):
            word = r[1].strip().split()[0]
            vector_dict[word] = r[0] + 1
    total_v = len(vector_dict.keys())
    return vector_dict

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return "nnumm"
    else: return word.lower()

def get_year(path):
    year_grab = re.compile("(\d\d\d\d).+")
    return int(year_grab.search(path).group(1))

def tokenize(read_doc):
    tokenized = nltk.tokenize.word_tokenize(read_doc.read().decode("utf-8"))
    tokenized = [normalize(word) for word in tokenized]
    return tokenized

def year_counter(year):
    global year_counts
    if year not in year_counts:
        year_counts[year] = 1
    elif year in year_counts:
        co = year_counts[year]
        if co == Config.per_year_cap:
            "print yay"
            return False
        else:
            co += 1
            year_counts[year] = co
            return True

def pickle(vector_to_pickle, year_bucket):
    dev, train = train_dev_divide(vector_to_pickle)
    print len(dev), "dev"
    print len(train), "train"
    pass
    with open(os.path.join(Config.pickle_dir, str(year_bucket) + "_train"), "wb") as dumping_file:
         cPickle.dump(train, dumping_file)
    with open(os.path.join(Config.pickle_dir, str(year_bucket) + "_dev"), "wb") as dumping_file:
         cPickle.dump(dev, dumping_file)

def train_dev_divide(vector_of_docs):
    counts_dev = int(len(vector_of_docs)*Config.percentage_dev)
    dev = vector_of_docs[-counts_dev:]
    train = vector_of_docs[:-counts_dev]
    return dev, train

def iterate_and_pickle(paths, files, vector_dict, year_buckets):
    all_docs_per_year = []
    prev_bucket = 0
    for path in zip(paths,files):
        with open(path[0], "rb") as read_doc:
            year = get_year(path[1])
            if year not in year_buckets:
                continue
            current_bucket = year_buckets[year]

            if prev_bucket < current_bucket:
                pickle(all_docs_per_year, prev_bucket)
                all_docs_per_year = []
                prev_bucket += 1
            tokenized = tokenize(read_doc)
            tok = len(tokenized)
            if tok <= 0:
                continue
            if not year_counter(year):
                continue
            ints = []
            for x in tokenized:
                if x in vector_dict:
                    ints.append(vector_dict[x])
                else:
                    ints.append(Config.not_a_word)
            vec_docs = make_50_long(ints)
            all_docs_per_year.extend(vec_docs)
    print year_counts
    pickle(all_docs_per_year, prev_bucket)


def make_50_long(tokenized):
    len_tokens = len(tokenized)
    count = 0
    return_tokenized = []
    while count + Config.doc_lengths < len_tokens:
        return_tokenized.append(tokenized[count:count+Config.doc_lengths])
        count += Config.doc_lengths
    return return_tokenized

if __name__ == "__main__":
    year_buckets = make_year_buckets()
    makedir()
    paths, files = return_paths()
    vector_dict = embeddings_dict()
    iterate_and_pickle(paths, files, vector_dict, year_buckets)


'''


'''


'''
{1890: 0, 1891: 0, 1892: 0, 1893: 0, 1894: 0, 1895: 1, 1896: 1, 1897: 1,
 1898: 1, 1899: 1, 1900: 2, 1901: 2, 1902: 2, 1903: 2, 1904: 2, 1905: 3,
 1906: 3, 1907: 3, 1908: 3, 1909: 3, 1910: 4, 1911: 4, 1912: 4, 1913: 4,
 1914: 4, 1915: 5, 1916: 5, 1917: 5, 1918: 5, 1919: 5, 1920: 6, 1921: 6,
 1922: 6, 1923: 6, 1924: 6, 1925: 7, 1926: 7, 1927: 7, 1928: 7, 1929: 7,
 1930: 8, 1931: 8, 1932: 8, 1933: 8, 1934: 8, 1935: 9, 1936: 9, 1937: 9,
 1938: 9, 1939: 9, 1940: 10, 1941: 10, 1942: 10, 1943: 10, 1944: 10,
 1945: 11, 1946: 11, 1947: 11, 1948: 11, 1949: 11, 1950: 12, 1951: 12,
 1952: 12, 1953: 12, 1954: 12, 1955: 13, 1956: 13, 1957: 13, 1958: 13,
 1959: 13, 1960: 14, 1961: 14, 1962: 14, 1963: 14, 1964: 14, 1965: 15,
 1966: 15, 1967: 15, 1968: 15, 1969: 15, 1970: 16, 1971: 16, 1972: 16,
 1973: 16, 1974: 16, 1975: 17, 1976: 17, 1977: 17, 1978: 17, 1979: 17}
'''

'''
3529 dev
20002 train
3799 dev
21533 train
2851 dev
16160 train
3820 dev
21653 train
4447 dev
25202 train
17275 dev
97892 train
7640 dev
43296 train
10128 dev
57393 train
16584 dev
93979 train
7457 dev
42258 train
26261 dev
148815 train
32964 dev
186799 train
45930 dev
260274 train
35506 dev
201206 train
34125 dev
193381 train
29237 dev
165681 train
35672 dev
202142 train
17200 dev
97467 train
'''