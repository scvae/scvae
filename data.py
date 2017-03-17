#!/usr/bin/env python3

import os
import sys
import urllib.request
import gzip
import pickle

from pandas import read_csv
from numpy import random, array, arange, zeros, sum, where, log, sort, clip #, nonzero, sort, argsort, where
from scipy.sparse import csr_matrix

from time import time
from auxiliary import convertTimeToString

preprocess_suffix = "preprocessed"
sparse_extension = ".pkl.gz"

data_set_URLs = {
    "mouse retina": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63472/suppl/GSE63472_P14Retina_merged_digital_expression.txt.gz",
}

class BaseDataSet(object):
    def __init__(self, counts = None, cells = None, genes = None, preprocessed_counts = None,
        name = "", kind = "", version = "original"):
        
        super(BaseDataSet, self).__init__()
        
        self.name = name
        self.kind = kind
        self.version = version
        
        self.update(counts, cells, genes, preprocessed_counts)
    
    def update(self, counts = None, cells = None, genes = None, preprocessed_counts = None):
        
        if counts is not None and cells is not None and genes is not None:
            
            M_counts, F_counts = counts.shape
            M_cells = cells.shape[0]
            F_genes = genes.shape[0]
            
            assert M_counts == M_cells
            assert F_counts == F_genes
            
            self.number_of_examples = M_cells
            self.number_of_features = F_genes
        
        else:
            self.number_of_examples = None
            self.number_of_features = None
        
        self.counts = counts
        self.preprocessed_counts = preprocessed_counts
        self.cells = cells
        self.genes = genes

class DataSet(BaseDataSet):
    def __init__(self, name, directory, preprocessing_method = None):
        super(DataSet, self).__init__(name = name, kind = "full")
        
        self.directory = directory
        
        if self.name != "sample":
            self.preprocess_directory = os.path.join(directory,
                preprocess_suffix)
            
            self.URL = data_set_URLs[self.name]
            
            file_name_with_extension = os.path.split(self.URL)[-1]
            
            file_name, extension = file_name_with_extension.split(os.extsep, 1)
            
            self.path = os.path.join(self.directory, file_name) + os.extsep + extension

            self.preprocessPath = lambda additions: \
                os.path.join(self.preprocess_directory, file_name) + "_" \
                    + "_".join(additions) + sparse_extension
            
            self.sparse_path = self.preprocessPath(["sparse"])
        
        self.load()
    
    def load(self):
        
        if self.name == "sample":
            data_dictionary = self.createSamples()
        else:
            if os.path.isfile(self.sparse_path):
                print("Loading data set from sparse representation.")
                start_time = time()
                data_dictionary = loadFromSparseData(self.sparse_path)
                duration = time() - start_time
                print("Data set loaded from sparse representation" +
                    " ({}).".format(convertTimeToString(duration)))
            else:
                if not os.path.isfile(self.path):
                    print("Downloading data set.")
                    start_time = time()
                    self.download()
                    duration = time() - start_time
                    print("Data set downloaded." +
                        " ({}).".format(convertTimeToString(duration)))
                
                print("Loading original data set.")
                start_time = time()
                data_dictionary = self.loadOriginalData()
                duration = time() - start_time
                print("Original data set loaded" +
                    " ({}).".format(convertTimeToString(duration)))
                
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving data set in sparse representation.")
                start_time = time()
                saveAsSparseData(data_dictionary, self.sparse_path)
                duration = time() - start_time
                print("Data set saved in sparse representation" +
                    " ({}).".format(convertTimeToString(duration)))
        
        self.update(
            data_dictionary["counts"],
            data_dictionary["cells"],
            data_dictionary["genes"]
        )
    
    def loadOriginalData(self):
        
        data_frame = read_csv(self.path, sep='\s+', index_col = 0,
            compression = "gzip", engine = "python", nrows = 5)
        
        data_dictionary = {
            "counts": data_frame.values.T,
            "cells": array(data_frame.columns.tolist()),
            "genes": array(data_frame.index.tolist())
        }
        
        return data_dictionary
    
    def download(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        urllib.request.urlretrieve(self.URL, self.path,
            download_report_hook)
    
    def createSamples(self, number_of_examples = 2000, number_of_features = 20,
        scale = 2, update_probability = 0.5):
        
        print("Creating sample data set.")
        
        random.seed(60)
        
        m = number_of_examples
        n = number_of_features
        
        samples = zeros((m, n))
        
        row = scale * random.rand(n)
        k = 0
        for i in range(m):
            u = random.rand()
            if u > 1 - update_probability:
                row = scale * random.rand(n)
                k += 1
            samples[i] = row
        
        random.shuffle(samples)
        
        for i in range(m):
            for j in range(n):
                samples[i, j] = random.poisson(samples[i, j])
        
        data_dictionary = {
            "counts": samples,
            "cells": arange(m),
            "genes": arange(n)
        }
        
        print("Sample data created with {} different example types.".format(k))
        
        return data_dictionary
    
    def split(self, method, fraction, preprocessing_method = None):
        
        if self.name == "sample":
            print("Splitting data set.")
            data_dictionary = self.splitAndCollectInDictionary(method,
                fraction, preprocessing_method)
            print("Data set split.")
        
        else:
            if preprocessing_method:
                split_data_sets_path = self.preprocessPath([
                    "split", preprocessing_method,
                    method, str(fraction)
                ])
            else:
                split_data_sets_path = self.preprocessPath([
                    "split",
                    method, str(fraction)
                ])
        
            if os.path.isfile(split_data_sets_path):
                print("Loading split data sets from sparse representations.")
                start_time = time()
                data_dictionary = loadFromSparseData(split_data_sets_path)
                duration = time() - start_time
                print("Split data sets loaded from sparse representations" +
                    " ({}).".format(convertTimeToString(duration)))
            else:
                print("Splitting data set.")
                start_time = time()
                data_dictionary = self.splitAndCollectInDictionary(method,
                    fraction, preprocessing_method)
                duration = time() - start_time
                print("Data set split" +
                    " ({}).".format(convertTimeToString(duration)))
                
                print("Saving split data sets in sparse representations.")
                start_time = time()
                saveAsSparseData(data_dictionary, split_data_sets_path)
                duration = time() - start_time
                print("Split data sets saved in sparse representations" +
                    " ({}).".format(convertTimeToString(duration)))
        
        training_set = BaseDataSet(
            counts = data_dictionary["training_set"]["counts"],
            preprocessed_counts = data_dictionary["training_set"]["counts"],
            cells = data_dictionary["training_set"]["cells"],
            genes = data_dictionary["genes"],
            name = self.name,
            kind = "training"
        )
        validation_set = BaseDataSet(
            counts = data_dictionary["validation_set"]["counts"],
            preprocessed_counts = data_dictionary["validation_set"]["counts"],
            cells = data_dictionary["validation_set"]["cells"],
            genes = data_dictionary["genes"],
            name = self.name,
            kind = "validation"
        )
        test_set = BaseDataSet(
            counts = data_dictionary["test_set"]["counts"],
            preprocessed_counts = data_dictionary["test_set"]["counts"],
            cells = data_dictionary["test_set"]["cells"],
            genes = data_dictionary["genes"],
            name = self.name,
            kind = "test"
        )
        
        print()
        
        print(
            "Data sets with {} features:\n".format(
                training_set.number_of_features) +
            "    Training sets: {} examples.\n".format(
                training_set.number_of_examples) +
            "    Validation sets: {} examples.\n".format(
                validation_set.number_of_examples) +
            "    Test sets: {} examples.".format(
                test_set.number_of_examples)
        )
        
        return training_set, validation_set, test_set
    
    def splitAndCollectInDictionary(self, method, fraction, preprocessing_method = None):
        
        random.seed(42)
        
        print("What preprocessing to use?")
        if preprocessing_method == "binarise":
            preprocess = lambda x: (x != 0).astype('float32')
        elif preprocessing_method == "gini":
            print("GINI preprocessing")
            preprocess = lambda x: x * gini_coefficients(x)
        elif preprocessing_method == "tf_idf":
            preprocess = lambda x: x * log_inverse_global_frequency(x)
        else:
            preprocess = lambda x: x
        
        self.preprocessed_counts = preprocess(self.counts)
        
        M = self.number_of_examples
        
        if method == "random":
            
            V = int(fraction * M)
            T = int(fraction * V)
            
            shuffled_indices = random.permutation(M)
            
            training_indices = shuffled_indices[:T]
            validation_indices = shuffled_indices[T:V]
            test_indices = shuffled_indices[V:]
        
        data_dictionary = {
            "training_set": {
                "counts": self.counts[training_indices],
                "preprocessed_counts": self.preprocessed_counts[training_indices],
                "cells": self.cells[training_indices],
            },
            "validation_set": {
                "counts": self.counts[validation_indices],
                "preprocessed_counts": self.preprocessed_counts[validation_indices],
                "cells": self.cells[validation_indices],
            },
            "test_set": {
                "counts": self.counts[test_indices],
                "preprocessed_counts": self.preprocessed_counts[test_indices],
                "cells": self.cells[test_indices],
            },
            "genes": self.genes
        }
        
        return data_dictionary

def loadFromSparseData(path):
    
    def converter(data):
        if data.ndim == 2:
            return data.todense().A
        else:
            return data
    
    with gzip.open(path, "rb") as data_file:
        data_dictionary = pickle.load(data_file)
    
    for key in data_dictionary:
        if "set" in key:
            for key2 in data_dictionary[key]:
                data_dictionary[key][key2] = converter(data_dictionary[key][key2])
        else:
            data_dictionary[key] = converter(data_dictionary[key])
    
    return data_dictionary

def saveAsSparseData(data_dictionary, path):
    
    sparse_data_dictionary = {}
    
    def converter(data):
        if data.ndim == 2:
            return csr_matrix(data)
        else:
            return data
    
    for key in data_dictionary:
        if "set" in key:
            sparse_data_dictionary[key] = {}
            for key2 in data_dictionary[key]:
                sparse_data_dictionary[key][key2] = \
                    converter(data_dictionary[key][key2])
        else:
            sparse_data_dictionary[key] = converter(data_dictionary[key])
    
    with gzip.open(path, "wb") as data_file:
        pickle.dump(sparse_data_dictionary, data_file)

def download_report_hook(block_num, block_size, total_size):
    bytes_read = block_num * block_size
    if total_size > 0:
        percent = bytes_read / total_size * 100
        sys.stderr.write("\rDownloading: {:3.0f}%.".format(percent))
        if bytes_read >= total_size:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("Downloaded {:d} bytes.".format(bytes_read))

### Prepocessing functions

# Compute gini-coefficients
def gini_coefficients(data, eps=1e-16, batch_size=5000):
    """Calculate the Gini coefficients along last axis of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    N, M = data.shape # number of examples, N, and features, M.
    index_vector =  2 * arange(1, N + 1) - N - 1  # 1-indexing vector for each data element.
    data = clip(data, eps, data) #values cannot be 0

    gini_index = zeros(M)
    print("Calculating gini-coefficients batch-wise for all features in the data set")
    for i in range(0, M, batch_size):
        batch = data[:, i:(i+batch_size)]

        # Array should be normalized and sorted frequencies over the examples
        batch = sort(batch / (sum(batch,axis=0)), axis=0)
        
        #Gini coefficients over the examples for each feature. 
        gini_index[i:(i+batch_size)] = index_vector @ batch / N

        print("Computed gini-indices for feature {} to {} out of {} features.".format(i, min(i+batch_size, M), M))

    return gini_index

# Compute log inverse global frequency of features over all examples.
def log_inverse_global_frequency(data):
    global_freq = sum(where(data > 0, 1, 0), axis=0)
    
    # Return log(inverse global frequency) = log(N / (global_frequency + 1))
    return log(data.shape[0] / (global_freq + 1))