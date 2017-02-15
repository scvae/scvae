#!/usr/bin/env python3

import gzip
import pickle
import os

from pandas import read_csv
from numpy import random, array#, zeros, nonzero, sort, argsort, where, arange
from scipy.sparse import csr_matrix

preprocess_suffix = "preprocessed"

data_sets = {
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
        
        if counts is not None and cells is not None and genes is not None:
            
            M_counts, F_counts = counts.shape
            M_cells = cells.shape[0]
            F_genes = genes.shape[0]
            
            assert M_counts == M_cells
            assert F_counts == F_genes
        
        self.counts = counts
        self.cells = cells
        self.genes = genes

class DataSet(BaseDataSet):
    def __init__(self, name, directory):
        super(DataSet, self).__init__()
        
        self.name = name
        self.directory = directory
        
        self.preprocess_directory = os.path.join(directory, preprocess_suffix)
        
        self.file_name = data_sets[self.name]["file name"]
        
        self.path = os.path.join(self.directory, self.file_name) \
            + data_sets[name]["extension"]
        
        self.preprocessPath = lambda additions: \
            os.path.join(self.preprocess_directory, self.file_name) \
                + "_" + "_".join(additions) + data_sets[name]["sparse extension"]
        
        self.sparse_path = self.preprocessPath(["sparse"])
        
        print(self.path)
        print(self.sparse_path)
        
        self.load()
    
    def load(self):
        if os.path.isfile(self.sparse_path):
            data_set = loadSparseDataSets(self.sparse_path)
        else:
            if not os.path.isfile(self.path):
                self.download()
            data_set = self.loadOriginalDataSet()
            if not os.path.exists(self.preprocess_directory):
                os.makedirs(self.preprocess_directory)
            saveSparseDataSets(data_set, self.sparse_path)
        
        self.counts = data_set.counts
        self.cells = data_set.cells
        self.genes = data_set.genes
    
    def loadOriginalDataSet(self):
        
        data_frame = read_csv(self.path, sep='\s+', index_col = 0,
            compression = "gzip", engine = "python")
        
        data_set = BaseDataSet(
            counts = data_frame.values.T,
            cells = array(data_frame.columns.tolist()),
            genes = array(data_frame.index.tolist()),
        )
        
        return data_set
    
    def download(self):
        pass
    
    def split(self, method, fraction):
        
        split_data_sets_path = self.preprocessPath(["split", method, str(fraction)])
        
        print(split_data_sets_path)
        
        if os.path.isfile(split_data_sets_path):
            training_set, validation_set, test_set = \
                 loadSparseDataSets(split_data_sets_path)
        else:
            training_set, validation_set, test_set = \
                self.splitIntoSets(method, fraction)
            saveSparseDataSets([training_set, validation_set, test_set],
                split_data_sets_path)
        
        return training_set, validation_set, test_set
    
    def splitIntoSets(self, method, fraction):
        
        M, F = self.counts.shape
        
        if method == "random":
            
            V = int(fraction * M)
            T = int(fraction * V)
            
            shuffled_indices = random.permutation(M)
            
            training_indices = shuffled_indices[:T]
            validation_indices = shuffled_indices[T:V]
            test_indices = shuffled_indices[V:]
        
        training_set = BaseDataSet(
            counts = self.counts[training_indices],
            cells = self.cells[training_indices],
            genes = self.genes
        )
        validation_set = BaseDataSet(
            counts = self.counts[validation_indices],
            cells = self.cells[validation_indices],
            genes = self.genes
        )
        test_set = BaseDataSet(
            counts = self.counts[test_indices],
            cells = self.cells[test_indices],
            genes = self.genes
        )
        
        return training_set, validation_set, test_set

def loadSparseDataSets(path):
    
    converter = lambda sparse_data: sparse_data.todense().A
    
    with gzip.open(path, "rb") as data_file:
        sparse_data_sets = pickle.load(data_file)
    
    data_sets = []
    
    for sparse_data_set in sparse_data_sets:
        data_set = BaseDataSet(
            counts = converter(sparse_data_set.counts),
            cells = sparse_data_set.cells,
            genes = sparse_data_set.genes
        )
        data_sets.append(data_set)
    
    if len(data_sets) == 1:
        data_sets = data_sets[0]
    
    return data_sets

def saveSparseDataSets(data_sets, path):
    
    converter = lambda data: csr_matrix(data)
    
    if type(data_sets) != list:
        data_sets = [data_sets]
    
    sparse_data_sets = []
    
    for data_set in data_sets:
        sparse_data_set = BaseDataSet(
            counts = converter(data_set.counts),
            cells = data_set.cells,
            genes = data_set.genes
        )
        sparse_data_sets.append(sparse_data_set)
    
    with gzip.open(path, "wb") as data_file:
        pickle.dump(sparse_data_sets, data_file)
