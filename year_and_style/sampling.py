import os
import re
import collections
import matplotlib.pyplot as plt
import sys, getopt
import random
from shutil import copyfile
from datetime import datetime
import numpy as np

"""
    python sampling.py -b 10 -n 230000 -d "/home/teamjo/stylometry/docs_by_date_raw_parse/" -t 25000
"""



def list_dates(directory_path):
	'''
	This function returns the year and the file path

	Arg: path to directory that contains all the text file_names
	Returns: a list of tuples where each tuple's first element 
				if the year of the doc and the second is the path to the file
	'''
	file_names = os.listdir(directory_path)
	date_pattern = re.compile(".?(\d\d\d\d).+")
	path_and_date_pair = []
	for f in file_names:
		date = date_pattern.match(f)
		date = date.group(1)
		path = os.path.join(directory_path, f)
		path_and_date_pair.append((date,path))
	return path_and_date_pair



def year_distribution(years):
	'''
	This function returns the distribution of docs frequency over the years.
	Rid all keys with values less than 10.


	Arg: list of all of the year instances of the docs
	Returns: year to count dictionary


	Counter({'1945': 10145, '1944': 8160, '1949': 7269, '1943': 7093, '1948': 6693, 
		'1946': 6512, '1951': 6350, '1954': 6052, '1941': 6036, '1947': 6002, '1918': 5541, 
		'1942': 5331, '1940': 5170, '1950': 4886, '1917': 4095, '1953': 4007, '1933': 3799, 
		'1932': 3795, '1934': 3649, '1935': 3539, '1952': 3530, '1915': 3525, '1919': 3525, 
		'1955': 3230, '1956': 3173, '1962': 3074, '1914': 2954, '1964': 2829, '1931': 2818, 
		'1957': 2784, '1916': 2759, '1929': 2753, '1961': 2739, '1958': 2676, '1967': 2639, 
		'1971': 2635, '1963': 2585, '1965': 2559, '1928': 2483, '1920': 2459, '1968': 2394, 
		'1972': 2384, '1959': 2339, '1970': 2274, '1960': 2248, '1930': 2166, '1936': 2118, 
		'1966': 2020, '1927': 1900, '1969': 1884, '1973': 1748, '1921': 1705, '1922': 1679, 
		'1923': 1618, '1912': 1501, '1926': 1445, '1975': 1378, '1895': 1368, '1925': 1353, 
		'1974': 1299, '1924': 1289, '1913': 1286, '1906': 1267, '1894': 1194, '1905': 1105, 
		'1977': 1087, '1898': 1086, '1900': 1083, '1976': 1043, '1902': 1018, '1907': 970, 
		'1904': 958, '1910': 948, '1896': 942, '1978': 906, '1893': 880, '1899': 843, '1911': 830,
		'1901': 810, '1908': 725, '1891': 722, '1897': 718, '1892': 710, '1903': 686, '1909': 642,
		'1979': 634, '1888': 629, '1865': 565, '1890': 515, '1980': 490, '1982': 471, '1886': 375, 
		'1939': 365, '1861': 348, '1889': 327, '1938': 233, '1937': 232, '1887': 173, '1981': 115, 
		'1885': 81, '1983': 25})

	'''


	set_years = set(years)
	print sorted(set_years)
	counter=collections.Counter(years)
	x = []
	y = []
	for k,v in counter.iteritems():
		if v < 10 or int(k) > 1983:
			continue
		x.append(k)
		y.append(v)
	plot_data(x,y)
	return counter



def plot_data(x, y):
	'''
	arg: data points x, y in vectors
	Plots data
	'''

	plt.plot(x, y, 'ro')
	plt.show()

years = {'1945': 10145, '1944': 8160, '1949': 7269, '1943': 7093, '1948': 6693,
        '1946': 6512, '1951': 6350, '1954': 6052, '1941': 6036, '1947': 6002, '1918': 5541,
        '1942': 5331, '1940': 5170, '1950': 4886, '1917': 4095, '1953': 4007, '1933': 3799,
        '1932': 3795, '1934': 3649, '1935': 3539, '1952': 3530, '1915': 3525, '1919': 3525,
        '1955': 3230, '1956': 3173, '1962': 3074, '1914': 2954, '1964': 2829, '1931': 2818,
        '1957': 2784, '1916': 2759, '1929': 2753, '1961': 2739, '1958': 2676, '1967': 2639,
        '1971': 2635, '1963': 2585, '1965': 2559, '1928': 2483, '1920': 2459, '1968': 2394,
        '1972': 2384, '1959': 2339, '1970': 2274, '1960': 2248, '1930': 2166, '1936': 2118,
        '1966': 2020, '1927': 1900, '1969': 1884, '1973': 1748, '1921': 1705, '1922': 1679,
        '1923': 1618, '1912': 1501, '1926': 1445, '1975': 1378, '1895': 1368, '1925': 1353,
        '1974': 1299, '1924': 1289, '1913': 1286, '1906': 1267, '1894': 1194, '1905': 1105,
        '1977': 1087, '1898': 1086, '1900': 1083, '1976': 1043, '1902': 1018, '1907': 970,
        '1904': 958, '1910': 948, '1896': 942, '1978': 906, '1893': 880, '1899': 843, '1911': 830,
        '1901': 810, '1908': 725, '1891': 722, '1897': 718, '1892': 710, '1903': 686, '1909': 642,
        '1979': 634, '1888': 629, '1865': 565, '1890': 515, '1980': 490, '1982': 471, '1886': 375,
        '1939': 365, '1861': 348, '1889': 327, '1938': 233, '1937': 232, '1887': 173, '1981': 115,
        '1885': 81, '1983': 25}

# Stratified sampling:
    # Arguments:
        # Bucket size: 5 years, 10 years.
        # Sampling Size: How many samples we want to extract
    # Steps:
        # 1. Create a new folder for the bucket size
        #    This will be our work space
        # 2. Calculate the probability distribution of each bucket.
        #    Use sampling size and probability distribution of each bucket
        # 3. For each bucket, create array of file names
        #    and put them in.
        # 4. Do the sampling frome each bucket using the array of files
        #    for each bucket


def stratify_sampling(bucket_size, num_samples, data_dir, threshold):
    bucket_size = int(bucket_size)
    num_samples = int(num_samples)

    # Step1
    output_dir = "./b{}_n{}_{:%Y%m%d_%H%M%S}".format(bucket_size, num_samples, datetime.now())
    print "Step1: creating output folder ", output_dir
    output_metadata = output_dir + "/output_metadata.txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        dev_dir = os.path.join(output_dir, "dev")
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(dev_dir)
    # Step2 Calculate the probability of each class
    #       Calcuate the number of samples per bucket
    print "Step2: Creating the number of samples for each bucket"
    start_date = 1861
    end_date = 1983
    num_buckets = (int(end_date) - int(start_date)) / bucket_size + 1
    print "     NUMBER OF BUCKETS: ", num_buckets
    bucket_counts = [0] * num_buckets   # number of documents per bucket
    total_documents = 0
    for year in years:
        # increment total documents count
        total_documents = total_documents + years[year]
        # calculate the bucket this year belongs to
        bucket_index = (int(year) - start_date) / bucket_size
        bucket_counts[bucket_index] = bucket_counts[bucket_index] + years[year]
    for i in range(len(bucket_counts)):
        bucket_counts[i] = int(bucket_counts[i]/(total_documents*1.0)*num_samples) # Convert to probability
        bucket_counts[i] = min(bucket_counts[i], threshold)


        print "     bucket ", i*bucket_size + start_date, "-", (i+1)*bucket_size + start_date, "have ", bucket_counts[i], "samples"
    print "     sample from each buckets: ", bucket_counts
    #Step3 Create a hash that maps bucket index to list of filenames for each bucket.
    #We will sample from


    print "Step3: Grouping filename into buckets"
    file_names = {}
    for filename in os.listdir(data_dir):
        file_year = int(filename[:4])
        bucket_index = (file_year - start_date) / bucket_size
        if bucket_index in file_names:
            file_names_value = file_names[bucket_index]
            file_names_value.append(filename)
            file_names[bucket_index] = file_names_value
            #file_names[bucket_index].append(filename)
        else:
            file_names[bucket_index] = [filename]

    # Step4 Use bucket_counts to sample from each bucket randomly.
    #       Move each chosen file from data dir to the output dir
    print "Step4: sample randomly from each bucket using precalculate counts"
    for i in range(len(bucket_counts)):
        if i not in file_names:   # No filenames for this bucket
            continue

        if bucket_counts[i] == 0:
            continue

        files = file_names[i]
        random.shuffle(files)

        num_files = bucket_counts[i]
        files = files[:num_files]

        # Take a sample from bucket
        num_train  = int(num_files * 0.7)
        num_dev = int(num_files * 0.15)

        for j in range(bucket_counts[i]):
            src = data_dir + files[j]
            if j <num_train:
                copyfile(src, os.path.join(train_dir, files[j]))
            elif j > num_train and j < num_train + num_dev:
                copyfile(src, os.path.join(dev_dir, files[j]))
            else:
                copyfile(src, os.path.join(test_dir, files[j]))
        ## train(70), test(15), dev(15)




def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hb:n:d:t:",["bucket_size=","num_samples=", "threshold="])
    except getopt.GetoptError:
        print 'test.py -b <years per bucket> -n <number of samples we want to extract> -d <training files dir> \
              -t <max num of docs per bucket>'
        sys.exit(2)

    bucket_size = None
    num_samples = None
    data_dir = None
    threshold = None
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -b <years per bucket> -n <number of samples we want to extract> -d <training files dir> \
                   -t <max num of docs per bucket>'
            sys.exit()
        elif opt in ("-b", "--bucket_size"):
            bucket_size = arg
        elif opt in ("-n", "--num_samples"):
            num_samples = arg
        elif opt in ("-d"):
            data_dir = arg
        elif opt in ("-t", "--threshold"):
            threshold = int(arg)


    if bucket_size is not None and num_samples is not None and data_dir is not None:
        #print "Performing stratified sampling over ", bucket_size, "year bucket size" , num_samples, "samples"
        stratify_sampling(bucket_size, num_samples, data_dir, threshold)
        sys.exit(0)

    # Default behavior
    directory = "/Users/eunseo/Desktop/frus/docs_by_date_raw_parse/"
    pairs = list_dates(directory)
    years, paths = zip(*pairs)
    year_distribution(years)

if __name__ == "__main__":
   main(sys.argv[1:])
