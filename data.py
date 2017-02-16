#!/usr/bin/env python3

import gzip
import pickle
import os

from pandas import read_csv
from numpy import random, array#, zeros, nonzero, sort, argsort, where, arange
from scipy.sparse import csr_matrix

preprocess_suffix = "preprocessed"

data_set_details = {
    "mouse retina": {
        "file name": "GSE63472_P14Retina_merged_digital_expression",
        "extension": ".txt.gz",
        "sparse extension": ".pkl.gz",
        "base_url": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63472/suppl/"
    }
}

class BaseDataSet(object):
    def __init__(self, counts = None, cells = None, genes = None):
        super(BaseDataSet, self).__init__()
        
        self.update(counts, cells, genes)
    
    def update(self, counts = None, cells = None, genes = None):
        
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
        self.cells = cells
        self.genes = genes

class DataSet(BaseDataSet):
    def __init__(self, name, directory):
        super(DataSet, self).__init__()
        
        self.name = name
        self.directory = directory
        
        self.preprocess_directory = os.path.join(directory, preprocess_suffix)
        
        self.file_name = data_set_details[self.name]["file name"]
        
        self.path = os.path.join(self.directory, self.file_name) \
            + data_set_details[name]["extension"]
        
        self.preprocessPath = lambda additions: \
            os.path.join(self.preprocess_directory, self.file_name) + "_" \
                + "_".join(additions) + data_set_details[name]["sparse extension"]
        
        self.sparse_path = self.preprocessPath(["sparse"])
        
        self.load()
    
    def load(self):
        if os.path.isfile(self.sparse_path):
            print(self.sparse_path)
            data_dictionary = loadFromSparseData(self.sparse_path)
        else:
            if not os.path.isfile(self.path):
                self.download()
            print(self.path)
            data_dictionary = self.loadOriginalData()
            if not os.path.exists(self.preprocess_directory):
                os.makedirs(self.preprocess_directory)
            print(self.sparse_path)
            saveAsSparseData(data_dictionary, self.sparse_path)
        
        self.update(
            data_dictionary["counts"],
            data_dictionary["cells"],
            data_dictionary["genes"]
        )
    
    def loadOriginalData(self):
        
        data_frame = read_csv(self.path, sep='\s+', index_col = 0,
            compression = "gzip", engine = "python")
        
        data_dictionary = {
            "counts": data_frame.values.T,
            "cells": array(data_frame.columns.tolist()),
            "genes": array(data_frame.index.tolist())
        }
        
        return data_dictionary
    
    def download(self):
        raise NotImplementedError
    
    def split(self, method, fraction):
        
        split_data_sets_path = self.preprocessPath(["split", method,
            str(fraction)])
        
        print(split_data_sets_path)
        
        if os.path.isfile(split_data_sets_path):
            data_dictionary = loadFromSparseData(split_data_sets_path)
        else:
            data_dictionary = self.splitAndCollectInDictionary(method,
                fraction)
            saveAsSparseData(data_dictionary, split_data_sets_path)
        
        training_set = BaseDataSet(
            counts = data_dictionary["training_set"]["counts"],
            cells = data_dictionary["training_set"]["cells"],
            genes = data_dictionary["genes"]
        )
        validation_set = BaseDataSet(
            counts = data_dictionary["validation_set"]["counts"],
            cells = data_dictionary["validation_set"]["cells"],
            genes = data_dictionary["genes"]
        )
        test_set = BaseDataSet(
            counts = data_dictionary["test_set"]["counts"],
            cells = data_dictionary["test_set"]["cells"],
            genes = data_dictionary["genes"]
        )
        
        return training_set, validation_set, test_set
    
    def splitAndCollectInDictionary(self, method, fraction):
        
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
                "cells": self.cells[training_indices],
            },
            "validation_set": {
                "counts": self.counts[validation_indices],
                "cells": self.cells[validation_indices],
            },
            "test_set": {
                "counts": self.counts[test_indices],
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
    
    def converter(data):
        if data.ndim == 2:
            return csr_matrix(data)
        else:
            return data
    
    for key in data_dictionary:
        if "set" in key:
            for key2 in data_dictionary[key]:
                data_dictionary[key][key2] = converter(data_dictionary[key][key2])
        else:
            data_dictionary[key] = converter(data_dictionary[key])
    
    with gzip.open(path, "wb") as data_file:
        pickle.dump(data_dictionary, data_file)
