#!/usr/bin/env python3

import os
import gzip
import tarfile
import pickle
import struct

import re
from bs4 import BeautifulSoup

import pandas
import numpy
import scipy.sparse
import sklearn.preprocessing
import stemming.porter2 as stemming

from functools import reduce

from time import time

from auxiliary import (
    formatDuration,
    normaliseString,
    download
)

from analysis import lighter_palette

preprocess_suffix = "preprocessed"
original_suffix = "original"
preprocessed_extension = ".sparse.pkl.gz"

data_sets = {
    "mouse retina": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "value": "transcript count"
        },
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "values": {
                "full": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63472/suppl/GSE63472_P14Retina_merged_digital_expression.txt.gz"
            },
            "labels": {
                "full": "http://mccarrolllab.com/wp-content/uploads/2015/05/retina_clusteridentities.txt"
            }
        },
        "load function": lambda x: loadMouseRetinaDataSet(x),
        "example type": "counts",
        "class palette": {
             0: (0., 0., 0.),
             1: (0.92, 0.24, 0.10),
             2: (0.89, 0.60, 0.14),
             3: (0.78, 0.71, 0.18),
             4: (0.80, 0.74, 0.16),
             5: (0.79, 0.76, 0.16),
             6: (0.81, 0.80, 0.18),
             7: (0.77, 0.79, 0.11),
             8: (0.77, 0.80, 0.16),
             9: (0.73, 0.78, 0.14),
            10: (0.71, 0.79, 0.15),
            11: (0.68, 0.78, 0.20),
            12: (0.65, 0.78, 0.15),
            13: (0.63, 0.79, 0.12),
            14: (0.63, 0.80, 0.17),
            15: (0.61, 0.78, 0.16),
            16: (0.57, 0.78, 0.14),
            17: (0.55, 0.78, 0.16),
            18: (0.53, 0.79, 0.14),
            19: (0.52, 0.80, 0.16),
            20: (0.47, 0.80, 0.17),
            21: (0.44, 0.80, 0.13),
            22: (0.42, 0.80, 0.16),
            23: (0.42, 0.79, 0.13),
            24: (0.12, 0.79, 0.72),
            25: (0.13, 0.64, 0.79),
            26: (0.00, 0.23, 0.88),
            27: (0.00, 0.24, 0.90),
            28: (0.13, 0.23, 0.89),
            29: (0.22, 0.23, 0.90),
            30: (0.33, 0.22, 0.87),
            31: (0.42, 0.23, 0.89),
            32: (0.53, 0.22, 0.87),
            33: (0.59, 0.24, 0.93),
            34: (0.74, 0.14, 0.67),
            35: (0.71, 0.13, 0.62),
            36: (0.74, 0.09, 0.55),
            37: (0.74, 0.08, 0.50),
            38: (0.73, 0.06, 0.44),
            39: (0.74, 0.06, 0.38),
        },
        "label superset": {
            "Horizontal": [1],
            "Retinal ganglion": [2],
            "Amacrine": [i for i in range(3, 24)],
            "Rods": [24],
            "Cones": [25],
            "Bipolar": [i for i in range(26, 34)],
            "Muller glia": [34],
            "Miscellaneous": [i for i in range(35, 40)],
            "No class": [0]
        },
        "literature probabilities": {
            "Horizontal": 0.5 / 100,
            "Retinal ganglion": 0.5 / 100,
            "Amacrine": 0.07,
            "Rods": 79.9 / 100,
            "Cones": 2.1 / 100,
            "Bipolar": 7.3 / 100,
            "Muller glia": 2.8 / 100,
            "Miscellaneous": 0
        },
        "excluded classes": [
            0
        ],
        "excluded superset classes": [
            "No class"
        ],
        "heat map normalisation": {
            "name": "Macosko",
            "label": lambda symbol:
                "$\log ({} / n \\times 10^{{4}} + 1)$".format(symbol),
            "function": lambda values, normalisation:
                numpy.log(values / normalisation * 1e4 + 1)
        },
        "PCA limits": {
            "full": {
                "PC1": {
                    "minimum": -250,
                    "maximum": 1750
                },
                "PC2": {
                    "minimum": -700,
                    "maximum":  700
                }
            },
            "training": {
                "PC1": {
                    "minimum": -250,
                    "maximum": 1750
                },
                "PC2": {
                    "minimum": -700,
                    "maximum":  700
                }
            },
            "validation": {
                "PC1": {
                    "minimum": -300,
                    "maximum":  800
                },
                "PC2": {
                    "minimum": -300,
                    "maximum":  500
                }
            },
            "test": {
                "PC1": {
                    "minimum": -200,
                    "maximum":  800
                },
                "PC2": {
                    "minimum": -400,
                    "maximum":  400
                }
            }
        }
    },
    
    "MNIST (original)": {
        "tags": {
            "example": "digit",
            "feature": "pixel",
            "value": "intensity count"
        },
        "split": ["training", "test"],
        "preprocessing methods": None,
        "URLs": {
            "values": {
                    "training":
                        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                    "test":
                        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
            },
            "labels": {
                    "training":
                        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                    "test":
                        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
            },
        },
        "load function": lambda x: loadMNISTDataSet(x),
        "maximum value": 255,
        "example type": "images",
        "feature dimensions": (28, 28)
    },
    
    "MNIST (normalised)": {
        "tags": {
            "example": "digit",
            "feature": "pixel",
            "value": "value"
        },
        "split": ["training", "validation", "test"],
        "preprocessing methods": None,
        "URLs": {
            "all": {
                    "full": "http://deeplearning.net/data/mnist/mnist.pkl.gz"
            }
        },
        "load function": lambda x: loadNormalisedMNISTDataSet(x),
        "maximum value": 1,
        "example type": "images",
        "feature dimensions": (28, 28)
    },
    
    "MNIST (binarised)": {
        "tags": {
            "example": "digit",
            "feature": "pixel",
            "value": "pixel value"
        },
        "split": ["training", "validation", "test"],
        "preprocessing methods": ["binarise"],
        "URLs": {
            "values": {
                    "training":
                        "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
                    "validation":
                        "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
                    "test":
                        "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat"
            },
            "labels": {
                    "training": None,
                    "validation": None,
                    "test": None
            },
        },
        "load function": lambda x: loadBinarisedMNISTDataSet(x),
        "maximum value": 1,
        "example type": "images",
        "feature dimensions": (28, 28)
    },
    
    "Reuters": {
        "tags": {
            "example": "document",
            "feature": "word",
            "value": "count"
        },
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full": "http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz"
            }
        },
        "load function": lambda x: loadReutersDataSet(x)
    },
    
    "20 Newsgroups": {
        "tags": {
            "example": "document",
            "feature": "word",
            "value": "count"
        },
        "split": ["training", "test"],
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full":
                    "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
            }
        },
        "load function": lambda x: load20NewsgroupsDataSet(x),
        "example type": "counts"
    },
    
    "blobs": {
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/s147246/datasets/blobs.pkl.gz"
            }
        },
        "load function": lambda x: loadSampleDataSet(x),
        "example type": "dummy"
    },
    
    "circles": {
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/s147246/datasets/circles.pkl.gz"
            }
        },
        "load function": lambda x: loadSampleDataSet(x),
        "example type": "dummy"
    },
    
    "moons": {
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/s147246/datasets/moons.pkl.gz"
            }
        },
        "load function": lambda x: loadSampleDataSet(x),
        "example type": "dummy"
    },
    
    "sample": {
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/s152421/data-sets/count_samples.pkl.gz"
            }
        },
        "load function": lambda x: loadSampleDataSet(x),
        "example type": "counts"
    },
    
    "sample (sparse)": {
        "split": False,
        "preprocessing methods": None,
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/s152421/data-sets/count_samples_sparse.pkl.gz"
            }
        },
        "load function": lambda x: loadSampleDataSet(x),
        "example type": "counts"
    },
    
    "development": {
        "split": False,
        "preprocessing methods": None,
        "URLs": {},
        "load function": lambda x: loadDevelopmentDataSet(
            number_of_examples = 10000,
            number_of_features = 5 * 5,
            scale = 10,
            update_probability = 0.0001
        ),
        # "example type": "images",
        "example type": "counts",
        "feature dimensions": (5, 5),
        "class palette": {
            0: (0, 0, 0),
            1: (1, 0, 0),
            2: (0, 1, 0),
            3: (0, 0, 1)
        },
        "label superset": {
            "Rods": [1, 2],
            "Cones": [3],
            "No class": [0]
        },
        "literature probabilities": {
            "Rods": 0.8,
            "Cones": 0.2
        },
        "excluded classes": [
            0
        ],
        "excluded superset classes": [
            "No class"
        ],
        "heat map normalisation": {
            "name": "Macosko",
            "label": lambda symbol:
                "$\log ({} / n \\times 10^{{4}} + 1)$".format(symbol),
            "function": lambda values, normalisation:
                numpy.log(values / normalisation * 1e4 + 1)
        },
        "PCA limits": {
            "full": {
                "PC1": {
                    "minimum": -500,
                    "maximum": 2000
                },
                "PC2": {
                    "minimum": -500,
                    "maximum": 1000
                }
            },
            "training": {
                "PC1": {
                    "minimum": -500,
                    "maximum": 2000
                },
                "PC2": {
                    "minimum": -500,
                    "maximum": 1000
                }
            },
            "validation": {
                "PC1": {
                    "minimum": -500,
                    "maximum": 1500
                },
                "PC2": {
                    "minimum": -500,
                    "maximum": 1000
                }
            },
            "test": {
                "PC1": {
                    "minimum": -500,
                    "maximum": 1500
                },
                "PC2": {
                    "minimum": -300,
                    "maximum":  700
                }
            }
        }
    }
}

class DataSet(object):
    def __init__(self, name,
        values = None,
        total_standard_deviations = None,
        explained_standard_deviations = None,
        preprocessed_values = None, binarised_values = None,
        labels = None, class_names = None,
        example_names = None, feature_names = None,
        feature_selection = [], feature_parameter = None, example_filter = [],
        preprocessing_methods = [], preprocessed = None,
        noisy_preprocessing_methods = [],
        kind = "full", version = "original",
        directory = "data"):
        
        super(DataSet, self).__init__()
        
        # Name of data set
        self.name = normaliseString(name)
        
        # Title (proper name) of data set
        self.title = dataSetTitle(self.name)
        
        # Tags (with names for examples, feature, and values) of data set
        self.tags = dataSetTags(self.title)
        
        # Example type for data set
        self.example_type = dataSetExampleType(self.title)
        
        # Maximum value of data set
        self.maximum_value = dataSetMaximumValue(self.title)
        
        # Discreteness
        self.discreteness = self.example_type == "counts" \
            or (self.maximum_value != None and self.maximum_value == 255)
        
        # Feature dimensions for data set
        self.feature_dimensions = dataSetFeatureDimensions(self.title)
        
        # Literature probabilities for data set
        self.literature_probabilities = dataSetLiteratureProbabilities(self.title)
        
        # Label super set for data set
        self.label_superset = dataSetLabelSuperset(self.title)
        self.superset_labels = None
        self.number_of_superset_classes = None
        
        # Label palette for data set
        self.class_palette = dataSetClassPalette(self.title)
        self.superset_class_palette = supersetClassPalette(
            self.class_palette, self.label_superset)
        
        # Excluded classes for data set
        self.excluded_classes = dataSetExcludedClasses(self.title)
        
        # Excluded classes for data set
        self.excluded_superset_classes = dataSetExcludedSupersetClasses(
            self.title)
        
        # Values and their names as well as labels in data set
        self.values = None
        self.total_standard_deviations = None
        self.explained_standard_deviations = None
        self.count_sum = None
        self.normalised_count_sum = None
        self.preprocessed_values = None
        self.binarised_values = None
        self.labels = None
        self.example_names = None
        self.feature_names = None
        self.class_names = None
        self.number_of_examples = None
        self.number_of_features = None
        self.number_of_classes = None
        self.update(
            values = values,
            total_standard_deviations = total_standard_deviations,
            explained_standard_deviations = explained_standard_deviations,
            preprocessed_values = preprocessed_values,
            binarised_values = binarised_values,
            labels = labels,
            example_names = example_names,
            feature_names = feature_names,
            class_names = class_names
        )
        
        # Feature selction
        self.feature_selection = feature_selection
        self.feature_parameter = feature_parameter
        
        # Example filterering
        self.example_filter = example_filter
        
        if example_filter:
            self.example_filter = example_filter[0]
            if len(example_filter) > 1:
                self.example_filter_parameters = example_filter[1:]
            else:
                self.example_filter_parameters = None
        else:
            self.example_filter = None
            self.example_filter_parameters = None
        
        # Preprocessing methods
        self.preprocessing_methods = preprocessing_methods
        
        if preprocessed is None:
            data_set_preprocessing_methods = \
                dataSetPreprocessingMethods(self.title)
            if data_set_preprocessing_methods:
                self.preprocessed = True
            else:
                self.preprocessed = False
        else:
            self.preprocessed = preprocessed
        
        if self.preprocessed:
            self.preprocessing_methods = data_set_preprocessing_methods
        
        # Kind of data set (full, training, validation, test)
        self.kind = kind
        
        # Split indices for training, validation, and test sets
        self.split_indices = None
        
        # Version of data set (original, reconstructed)
        self.version = version
        
        # Heat map normalisation for data set
        self.heat_map_normalisation = dataSetHeatMapNormalisation(self.title)
        
        # PCA limits for data set
        self.pca_limits = dataSetPCALimits(self.title, self.kind)
        
        # Directories for data set
        self.directory = os.path.join(directory, self.name)
        self.preprocess_directory = os.path.join(self.directory,
            preprocess_suffix)
        self.original_directory = os.path.join(self.directory,
            original_suffix)
        
        self.preprocessedPath = preprocessedPathFunction(
            self.preprocess_directory, self.name)
        
        # Noisy preprocessing
        self.noisy_preprocessing_methods = noisy_preprocessing_methods
        
        if self.preprocessed:
            self.noisy_preprocessing_methods = []
        
        if self.noisy_preprocessing_methods:
            self.noisy_preprocess = preprocessingFunctionForDataSet(
                self.title, self.noisy_preprocessing_methods,
                self.preprocessedPath,
                noisy = True
            )
        else:
            self.noisy_preprocess = None
        
        if self.kind == "full":
            
            print("Data set:")
            print("    title:", self.title)
            
            if self.feature_selection:
                print("    feature selection:", self.feature_selection)
                if self.feature_parameter:
                    print("        parameter:", self.feature_parameter)
                else:
                    print("        parameter: default")
            else:
                print("    feature selection: none")
            
            if self.example_filter:
                print("    example filter:", self.example_filter)
                if self.example_filter_parameters:
                    print("        parameter(s):",
                        ", ".join(self.example_filter_parameters))
            else:
                print("    example filter: none")
            
            if not self.preprocessed and self.preprocessing_methods:
                print("    processing methods:")
                for preprocessing_method in self.preprocessing_methods:
                    print("        ", preprocessing_method)
            elif self.preprocessed:
                print("    processing methods: already done")
            else:
                print("    processing methods: none")
            
            if not self.preprocessed and self.noisy_preprocessing_methods:
                print("    noisy processing methods:")
                for preprocessing_method in self.noisy_preprocessing_methods:
                    print("        ", preprocessing_method)
            print()
            
            if self.values is None:
                self.load()
            
            if not self.feature_parameter:
                self.feature_parameter = self.defaultFeatureParameter(
                    self.feature_selection)
            
            if self.preprocessed_values is None:
                self.preprocess()
                if self.noisy_preprocessing_methods and \
                    self.noisy_preprocessing_methods[-1] != "binarise":
                    pass
                else:
                    self.binarise()
    
    @property
    def number_of_values(self):
        return self.number_of_examples * self.number_of_features
    
    @property
    def class_probabilities(self):
        
        if self.label_superset:
            labels = self.superset_labels
            class_names = self.superset_class_names
            excluded_classes = self.excluded_superset_classes
        else:
            labels = self.labels
            class_names = self.class_names
            excluded_classes = self.excluded_classes
        
        class_probabilities = {class_name: 0 for class_name in class_names}
        
        total_count_sum = 0
    
        for label in labels:
            if label in excluded_classes:
                continue
            class_probabilities[label] += 1
            total_count_sum += 1
        
        class_names_with_zero_probability = []
        
        for name, count in class_probabilities.items():
            if count == 0:
                class_names_with_zero_probability.append(name)
            class_probabilities[name] = count / total_count_sum
        
        for name in class_names_with_zero_probability:
            class_probabilities.pop(name)
        
        return class_probabilities
    
    def update(self, values = None,
        total_standard_deviations = None,
        explained_standard_deviations = None,
        preprocessed_values = None,
        binarised_values = None, labels = None,
        example_names = None, feature_names = None, class_names = None):
        
        if values is not None:
            
            self.values = values
            
            self.count_sum = self.values.sum(axis = 1).reshape(-1, 1)
            self.normalised_count_sum = self.count_sum / self.count_sum.max()
            
            M_values, N_values = values.shape
            
            if example_names is not None:
                self.example_names = example_names
                M_examples = example_names.shape[0]
                assert M_values == M_examples
            
            if feature_names is not None:
                self.feature_names = feature_names
                N_features = feature_names.shape[0]
                assert N_values == N_features
            
            self.number_of_examples = M_values
            self.number_of_features = N_values
        
        else:
            
            if example_names is not None and feature_names is not None:
                
                self.example_names = example_names
                self.feature_names = feature_names
        
        if labels is not None:
            
            self.labels = labels
            
            if class_names is not None:
                self.class_names = class_names
            else:
                self.class_names = numpy.unique(self.labels).tolist()
            
            self.class_id_to_class_name = {}
            self.class_name_to_class_id = {}
            
            for i, class_name in enumerate(self.class_names):
                self.class_name_to_class_id[class_name] = i
                self.class_id_to_class_name[i] = class_name
            
            self.number_of_classes = len(self.class_names)
            
            if self.label_superset:
                
                self.superset_labels = supersetLabels(
                    self.labels, self.label_superset)
                
                self.superset_class_names = sorted(self.label_superset.keys())
                
                self.superset_class_id_to_superset_class_name = {}
                self.superset_class_name_to_superset_class_id = {}
            
                for i, class_name in \
                    enumerate(self.superset_class_names):
                    self.superset_class_name_to_superset_class_id[class_name] = i
                    self.superset_class_id_to_superset_class_name[i] = class_name
                
                self.number_of_superset_classes = len(self.superset_class_names)
        
        if total_standard_deviations is not None:
            self.total_standard_deviations = total_standard_deviations
        
        if explained_standard_deviations is not None:
            self.explained_standard_deviations = explained_standard_deviations
        
        if preprocessed_values is not None:
            self.preprocessed_values = preprocessed_values
        
        if binarised_values is not None:
            self.binarised_values = binarised_values
    
    def load(self):
        
        sparse_path = self.preprocessedPath()
        
        if os.path.isfile(sparse_path):
            print("Loading data set from sparse representation.")
            data_dictionary = loadFromSparseData(sparse_path)
        else:
            original_paths = downloadDataSet(self.title, self.original_directory)
            
            print()
            
            data_dictionary, duration = loadOriginalDataSet(self.title, original_paths)
            
            print()
            
            if duration > 30:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving data set in sparse representation.")
                saveAsSparseData(data_dictionary, sparse_path)
        
        self.update(
            values = data_dictionary["values"],
            labels = data_dictionary["labels"],
            example_names = data_dictionary["example names"],
            feature_names = data_dictionary["feature names"]
        )
        
        if "split indices" in data_dictionary:
            self.split_indices = data_dictionary["split indices"]
        
        print()
    
    def defaultFeatureParameter(self, feature_selection):
        
        M = self.number_of_features
        
        if feature_selection:
            feature_selection = normaliseString(feature_selection)
        
        if feature_selection == "remove_zeros":
            feature_parameter = None
        
        elif feature_selection == "keep_gini_indices_above":
            feature_parameter = 0.1
        
        elif feature_selection == "keep_highest_gini_indices":
            feature_parameter = int(M/2)
        
        elif feature_selection == "keep_variances_above":
            feature_parameter = 0.5
        
        elif feature_selection == "keep_highest_variances":
            feature_parameter = int(M/2)
        
        else:
            feature_parameter = None
    
        return feature_parameter
        
    def preprocess(self):
        
        if not self.preprocessing_methods and not self.feature_selection \
            and not self.example_filter:
            self.update(preprocessed_values = self.values)
            return
        
        sparse_path = self.preprocessedPath(
            preprocessing_methods = self.preprocessing_methods,
            feature_selection = self.feature_selection, 
            feature_parameter = self.feature_parameter,
            example_filter = self.example_filter,
            example_filter_parameters = self.example_filter_parameters
        )
        
        if os.path.isfile(sparse_path):
            print("Loading preprocessed data from sparse representation.")
            data_dictionary = loadFromSparseData(sparse_path)
            data_dictionary["preprocessed values"] = self.values
        else:
            if not self.preprocessed and self.preprocessing_methods:
                    
                print("Preprocessing values.")
                start_time = time()
                
                preprocessing_function = preprocessingFunctionForDataSet(
                    self.title, self.preprocessing_methods, self.preprocessedPath)
                preprocessed_values = preprocessing_function(self.values)
                
                duration = time() - start_time
                print("Values preprocessed ({}).".format(formatDuration(duration)))
                
                print()
            
            else:
                preprocessed_values = self.values
            
            values = self.values
            example_names = self.example_names
            feature_names = self.feature_names
            
            if self.feature_selection:
                values_dictionary, feature_names = selectFeatures(
                    {"original": values,
                     "preprocessed": preprocessed_values},
                    self.feature_names,
                    self.feature_selection,
                    self.feature_parameter,
                    self.preprocessedPath
                )
                
                values = values_dictionary["original"]
                preprocessed_values = values_dictionary["preprocessed"]
            
                print()
                
            if self.example_filter:
                values_dictionary, example_names, labels = filterExamples(
                    {"original": values,
                     "preprocessed": preprocessed_values},
                    self.example_names,
                    self.example_filter,
                    self.example_filter_parameters,
                    labels = self.labels,
                    excluded_classes = self.excluded_classes,
                    superset_labels = self.superset_labels,
                    excluded_superset_classes = self.excluded_superset_classes,
                    count_sum = self.count_sum
                )
                
                values = values_dictionary["original"]
                preprocessed_values = values_dictionary["preprocessed"]
            
                print()
            
            data_dictionary = {
                "values": values,
                "preprocessed values": preprocessed_values,
            }
            
            if self.feature_selection:
                data_dictionary["feature names"] = feature_names
            
            if self.example_filter:
                data_dictionary["example names"] = example_names
                data_dictionary["labels"] = labels
            
            if self.number_of_features > 1000:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
            
                print("Saving preprocessed data set in sparse representation.")
                saveAsSparseData(data_dictionary, sparse_path)
        
        values = data_dictionary["values"]
        preprocessed_values = data_dictionary["preprocessed values"]
        
        if self.feature_selection:
            feature_names = data_dictionary["feature names"]
        else:
            feature_names = self.feature_names
        
        if self.example_filter:
            example_names = data_dictionary["example names"]
            labels = data_dictionary["labels"]
        else:
            example_names = self.example_names
            labels = self.labels
        
        self.update(
            values = values,
            preprocessed_values = preprocessed_values,
            feature_names = feature_names,
            example_names = example_names,
            labels = labels
        )
        
        print()
    
    def binarise(self):
        
        if self.preprocessed_values is None:
            raise NotImplementedError("Data set values have to have been",
                "preprocessed and feature selected first.")
        
        binarise_preprocessing = ["binarise"]
        
        sparse_path = self.preprocessedPath(
            preprocessing_methods = binarise_preprocessing,
            feature_selection = self.feature_selection, 
            feature_parameter = self.feature_parameter,
            example_filter = self.example_filter,
            example_filter_parameters = self.example_filter_parameters
        )
        
        if os.path.isfile(sparse_path):
            print("Loading binarised data from sparse representation.")
            data_dictionary = loadFromSparseData(sparse_path)
        
        else:
            if self.preprocessing_methods != binarise_preprocessing:
                
                print("Binarising values.")
                start_time = time()
                
                binarisation_function = preprocessingFunctionForDataSet(
                    self.title, binarise_preprocessing, self.preprocessedPath)
                binarised_values = binarisation_function(self.values)
                
                duration = time() - start_time
                print("Values binarised ({}).".format(formatDuration(duration)))
                
                print()
            
            elif self.preprocessing_methods == binarise_preprocessing:
                binarised_values = self.preprocessed_values
            
            data_dictionary = {
                "values": self.values,
                "preprocessed values": binarised_values,
                "feature names": self.feature_names
            }
            
            if self.number_of_features > 1000:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving binarised data set in sparse representation.")
                saveAsSparseData(data_dictionary, sparse_path)
        
        self.update(
            binarised_values = data_dictionary["preprocessed values"],
        )
        
        print()
    
    def defaultSplittingMethod(self):
        if self.split_indices:
            return "indices"
        else:
            return "random"
    
    def split(self, method = "default", fraction = 0.9):
        
        if method == "default":
            method = self.defaultSplittingMethod()
        
        sparse_path = self.preprocessedPath(
            preprocessing_methods = self.preprocessing_methods,
            feature_selection = self.feature_selection,
            feature_parameter = self.feature_parameter,
            example_filter = self.example_filter,
            example_filter_parameters = self.example_filter_parameters,
            splitting_method = method,
            splitting_fraction = fraction
        )
        
        print("Splitting:")
        print("    method:", method)
        if method != "indices":
            print("    fraction: {:.1f} %".format(100 * fraction))
        print()
        
        if os.path.isfile(sparse_path):
            print("Loading split data sets from sparse representation.")
            split_data_dictionary = loadFromSparseData(sparse_path)
        
        else:
            
            data_dictionary = {
                "values": self.values,
                "preprocessed values": self.preprocessed_values,
                "binarised values": self.binarised_values,
                "labels": self.labels,
                "example names": self.example_names,
                "split indices": self.split_indices
            }
            
            split_data_dictionary = splitDataSet(data_dictionary, method,
                fraction)
            
            print()
            
            if self.number_of_features > 1000:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving split data sets in sparse representation.")
                saveAsSparseData(split_data_dictionary, sparse_path)
        
        training_set = DataSet(
            name = self.name,
            values = split_data_dictionary["training set"]["values"],
            preprocessed_values = \
                split_data_dictionary["training set"]["preprocessed values"],
            binarised_values = \
                split_data_dictionary["training set"]["binarised values"],
            labels = split_data_dictionary["training set"]["labels"],
            example_names = split_data_dictionary["training set"]["example names"],
            feature_names = self.feature_names,
            class_names = self.class_names,
            feature_selection = self.feature_selection,
            example_filter = self.example_filter,
            preprocessing_methods = self.preprocessing_methods,
            noisy_preprocessing_methods = self.noisy_preprocessing_methods,
            kind = "training"
        )
        
        validation_set = DataSet(
            name = self.name,
            values = split_data_dictionary["validation set"]["values"],
            preprocessed_values = \
                split_data_dictionary["validation set"]["preprocessed values"],
            binarised_values = \
                split_data_dictionary["validation set"]["binarised values"],
            labels = split_data_dictionary["validation set"]["labels"],
            example_names = split_data_dictionary["validation set"]["example names"],
            feature_names = self.feature_names,
            class_names = self.class_names,
            feature_selection = self.feature_selection,
            example_filter = self.example_filter,
            preprocessing_methods = self.preprocessing_methods,
            noisy_preprocessing_methods = self.noisy_preprocessing_methods,
            kind = "validation"
        )
        
        test_set = DataSet(
            name = self.name,
            values = split_data_dictionary["test set"]["values"],
            preprocessed_values = \
                split_data_dictionary["test set"]["preprocessed values"],
            binarised_values = \
                split_data_dictionary["test set"]["binarised values"],
            labels = split_data_dictionary["test set"]["labels"],
            example_names = split_data_dictionary["test set"]["example names"],
            feature_names = self.feature_names,
            class_names = self.class_names,
            feature_selection = self.feature_selection,
            example_filter = self.example_filter,
            preprocessing_methods = self.preprocessing_methods,
            noisy_preprocessing_methods = self.noisy_preprocessing_methods,
            kind = "test"
        )
        
        print()
        
        print(
            "Data sets with {} features{}:\n".format(
                training_set.number_of_features,
                " and {} classes".format(self.number_of_classes)
                    if self.number_of_classes else "") +
            "    Training sets: {} examples.\n".format(
                training_set.number_of_examples) +
            "    Validation sets: {} examples.\n".format(
                validation_set.number_of_examples) +
            "    Test sets: {} examples.".format(
                test_set.number_of_examples)
        )
        
        print()
        
        return training_set, validation_set, test_set
    
    def applyIndices(self, indices):
        filter_indices = numpy.arange(self.number_of_examples)
        filter_indices = filter_indices[indices]
        self.update(
            values = self.values[filter_indices],
            labels = self.labels[filter_indices],
            example_names = self.example_names[filter_indices],
            feature_names = self.feature_names,
            class_names = self.class_names
        )
        if self.total_standard_deviations is not None:
            self.update(
                total_standard_deviations = \
                    self.total_standard_deviations[filter_indices])
        if self.explained_standard_deviations is not None:
            self.update(
                explained_standard_deviations = \
                    self.explained_standard_deviations[filter_indices])
        if self.preprocessed_values is not None:
            self.update(
                preprocessed_values = self.preprocessed_values[filter_indices])
        if self.binarised_values is not None:
            self.update(
                binarised_values = self.binarised_values[filter_indices])

def dataSetTitle(name):
    
    title = None
        
    for data_set in data_sets:
        if normaliseString(data_set) == name:
            title = data_set
    
    if not title:
        raise KeyError("Data set not found.")
    
    return title

def dataSetTags(title):
    if "tags" in data_sets[title]:
        return data_sets[title]["tags"]
    else:
        tags = {
            "example": "example",
            "feature": "feature",
            "value": "value"
        }
        return tags

def dataSetExampleType(title):
    if "example type" in data_sets[title]:
        return data_sets[title]["example type"]
    else:
        return None

def dataSetMaximumValue(title):
    if "maximum value" in data_sets[title]:
        return data_sets[title]["maximum value"]
    else:
        return None

def dataSetFeatureDimensions(title):
    if "feature dimensions" in data_sets[title]:
        return data_sets[title]["feature dimensions"]
    else:
        return None

def dataSetHeatMapNormalisation(title):
    if "heat map normalisation" in data_sets[title]:
        return data_sets[title]["heat map normalisation"]
    else:
        return None

def dataSetPCALimits(title, kind):
    if "PCA limits" in data_sets[title]:
        return data_sets[title]["PCA limits"][kind]
    else:
        return None

def dataSetLiteratureProbabilities(title):
    if "literature probabilities" in data_sets[title]:
        return data_sets[title]["literature probabilities"]
    else:
        return None

def dataSetClassPalette(title):
    if "class palette" in data_sets[title]:
        class_palette = data_sets[title]["class palette"]
    elif "MNIST" in title:
        index_palette = lighter_palette(10)
        class_palette = {i: index_palette[i] for i in range(10)}
    else:
        class_palette = None
    return class_palette

def dataSetLabelSuperset(title):
    if "label superset" in data_sets[title]:
        return data_sets[title]["label superset"]
    else:
        return None

def dataSetExcludedClasses(title):
    if "excluded classes" in data_sets[title]:
        return data_sets[title]["excluded classes"]
    else:
        return []

def dataSetExcludedSupersetClasses(title):
    if "excluded superset classes" in data_sets[title]:
        return data_sets[title]["excluded superset classes"]
    else:
        return []

def dataSetPreprocessingMethods(title):
    return data_sets[title]["preprocessing methods"]

def downloadDataSet(title, directory):
    
    URLs = data_sets[title]["URLs"]
    
    paths = {}
    
    if not URLs:
        return paths
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for values_or_labels in URLs:
        paths[values_or_labels] = {}
        
        for kind in URLs[values_or_labels]:
            
            URL = URLs[values_or_labels][kind]
            
            if not URL:
                continue
            
            URL_filename = os.path.split(URL)[-1]
            extension = os.extsep + URL_filename.split(os.extsep, 1)[-1]
            
            name = normaliseString(title)
            filename = name + "-" + values_or_labels + "-" + kind
            path = os.path.join(directory, filename) + extension
            
            paths[values_or_labels][kind] = path
            
            if not os.path.isfile(path):
                
                print("Downloading {} for {} set.".format(
                    values_or_labels, kind, title))
                start_time = time()
                
                download(URL, path)
                
                duration = time() - start_time
                print("Data set downloaded ({}).".format(formatDuration(duration)))
                
                print()
    
    return paths

def loadOriginalDataSet(title, paths):
    print("Loading original data set.")
    start_time = time()
    
    data_dictionary = data_sets[title]["load function"](paths)
    
    duration = time() - start_time
    print("Original data set loaded ({}).".format(formatDuration(duration)))
    
    return data_dictionary, duration

def preprocessedPathFunction(preprocess_directory = "", name = ""):
    
    def preprocessedPath(base_name = None, preprocessing_methods = None,
        feature_selection = None, feature_parameter = None,
        example_filter = None, example_filter_parameters = None,
        splitting_method = None, splitting_fraction = None):
        
        base_path = os.path.join(preprocess_directory, name)
        
        filename_parts = [base_path]
        
        if base_name:
            filename_parts.append(normaliseString(base_name))
        
        if feature_selection:
            filename_parts.append(normaliseString(feature_selection))
            if feature_parameter:
                filename_parts.append(str(feature_parameter))
        
        if example_filter:
            filename_parts.append(normaliseString(example_filter))
            if example_filter_parameters:
                filename_parts.extend(map(normaliseString,
                    example_filter_parameters))
        
        if preprocessing_methods:
            filename_parts.extend(map(normaliseString, preprocessing_methods))
        
        if splitting_method:
            filename_parts.append("split")
            filename_parts.append(normaliseString(splitting_method))
            
            if splitting_fraction:
                filename_parts.append(str(splitting_fraction))
            
        path = "-".join(filename_parts) + preprocessed_extension
        
        return path
    
    return preprocessedPath

def selectFeatures(values_dictionary, feature_names, feature_selection = None,
    feature_parameter = None, preprocessPath = None):
    
    feature_selection = normaliseString(feature_selection)
    
    print("Selecting features.")
    start_time = time()
    
    if type(values_dictionary) == dict:
        values = values_dictionary["original"]
    
    M, N = values.shape
    
    if feature_selection == "remove_zeros":
        total_feature_sum = values.sum(axis = 0)
        indices = total_feature_sum != 0
    
    elif feature_selection == "keep_gini_indices_above":
        gini_indices = loadWeights(values, "gini", preprocessPath)
        if not feature_parameter:
            feature_parameter = 0.1
        indices = gini_indices > feature_parameter

    elif feature_selection == "keep_highest_gini_indices":
        gini_indices = loadWeights(values, "gini", preprocessPath)
        gini_sorted_indices = numpy.argsort(gini_indices)
        if feature_parameter:
            feature_parameter = int(feature_parameter)
        else:
            feature_parameter = int(M/2)
        indices = numpy.sort(gini_sorted_indices[-feature_parameter:])
        
    elif feature_selection == "keep_variances_above":
        variances = values.var(axis = 0)
        if not feature_parameter:
            feature_parameter = 0.5
        indices = variances > feature_parameter

    elif feature_selection == "keep_highest_variances":
        variances = values.var(axis = 0)
        variance_sorted_indices = numpy.argsort(variances)
        if feature_parameter:
            feature_parameter = int(feature_parameter)
        else:
            feature_parameter = int(M/2)
        indices = numpy.sort(variance_sorted_indices[-feature_parameter:])
        
    else:
        indices = numpy.arange(N)
    
    feature_selected_values = {}
    
    for version, values in values_dictionary.items():
        feature_selected_values[version] = values[:, indices]
    
    feature_selected_feature_names = feature_names[indices]
    
    duration = time() - start_time
    print("{} features selected ({}).".format(
        len(indices),
        formatDuration(duration)
    ))
    
    return feature_selected_values, feature_selected_feature_names

def filterExamples(values_dictionary, example_names, example_filter = None,
    example_filter_parameters = None, labels = None, excluded_classes = None,
    superset_labels = None, excluded_superset_classes = None,
    count_sum = None):
    
    print("Filtering examples.")
    start_time = time()
    
    example_filter = normaliseString(example_filter)
    
    if superset_labels is not None:
        filter_labels = superset_labels.copy()
        filter_excluded_classes = excluded_superset_classes
    else:
        filter_labels = labels.copy()
        filter_excluded_classes = excluded_classes
    
    filter_class_names = numpy.unique(filter_labels)
    
    if type(values_dictionary) == dict:
        values = values_dictionary["original"]
    
    M, N = values.shape
    
    filter_indices = numpy.arange(M)
    
    if example_filter == "macosko":
        minimum_number_of_non_zero_elements = 900
        number_of_non_zero_elements = (values != 0).sum(axis = 1)
        filter_indices = numpy.nonzero(
            number_of_non_zero_elements > minimum_number_of_non_zero_elements
        )[0]
    
    elif example_filter in ["keep", "remove", "excluded_classes"]:
        
        if example_filter == "excluded_classes":
            example_filter = "remove"
            example_filter_parameters = filter_excluded_classes
        
        if example_filter == "keep":
            label_indices = set()
            
            for parameter in example_filter_parameters:
                for class_name in filter_class_names:
                    if normaliseString(str(class_name)) \
                        == normaliseString(str(parameter)):
                        
                        class_indices = filter_labels == class_name
                        label_indices.update(filter_indices[class_indices])

            filter_indices = filter_indices[list(label_indices)]
            
        elif example_filter == "remove":
            for parameter in example_filter_parameters:
                for class_name in filter_class_names:
                    if normaliseString(str(class_name)) \
                        == normaliseString(str(parameter)):
                        
                        label_indices = filter_labels != class_name
                        filter_labels = filter_labels[label_indices]
                        filter_indices = filter_indices[label_indices]
        
        
    
    elif example_filter == "remove_count_sum_above":
        threshold = int(example_filter_parameters[0])
        filter_indices = filter_indices[count_sum.reshape(-1) <= threshold]
    
    example_filtered_values = {}
    
    for version, values in values_dictionary.items():
        example_filtered_values[version] = values[filter_indices, :]
    
    example_filtered_example_names = example_names[filter_indices]
    example_filtered_labels = labels[filter_indices]
    
    duration = time() - start_time
    print("{} examples filtered out, {} remaining ({}).".format(
        M - len(filter_indices),
        len(filter_indices),
        formatDuration(duration)
    ))
    
    return example_filtered_values, example_filtered_example_names, \
        example_filtered_labels

def normalisationFunctionForDataSet(title):
    if "maximum value" in data_sets[title]:
        maximum_value = data_sets[title]["maximum value"]
        normalisation_function = lambda values: values / maximum_value
        if not "original maximum value" in data_sets[title]:
            data_sets[title]["original maximum value"] = maximum_value
        data_sets[title]["maximum value"] = 1
    else:
        normalisation_function = lambda values: sklearn.preprocessing.normalize(
            values, norm = 'l2', axis = 0)
    return normalisation_function

def bernoulliSample(p):
    return numpy.random.binomial(1, p)

def binarisationFunctionForDataSet(title, noisy = False):
    if "maximum value" in data_sets[title]:
        normalisation = data_sets[title]["maximum value"]
    else:
        normalisation = 1
    
    if noisy:
        binarisation_function = lambda values: bernoulliSample(
            values / normalisation)
    else:
        binarisation_function = lambda values: sklearn.preprocessing.binarize(
            values / normalisation, threshold = 0.5)
    
    return binarisation_function

def preprocessingFunctionForDataSet(title, preprocessing_methods = [],
    preprocessPath = None, noisy = False):
    
    preprocesses = []
    
    for preprocessing_method in preprocessing_methods:
        
        if preprocessing_method in ["gini", "idf"]:
            weight_method = preprocessing_method
            preprocess = lambda x: applyWeights(x, weight_method,
                preprocessPath)
        
        elif preprocessing_method == "normalise":
            preprocess = normalisationFunctionForDataSet(title)
        
        elif preprocessing_method == "binarise":
            preprocess = binarisationFunctionForDataSet(title, noisy)
        
        else:
            preprocess = lambda x: x
        
        preprocesses.append(preprocess)
    
    if not preprocessing_methods:
        preprocesses.append(lambda x: x)
    
    preprocessing_function = lambda x: reduce(
        lambda v, p: p(v),
        preprocesses,
        x
    )
    
    if "original maximum value" in data_sets[title]:
        data_sets[title]["maximum value"] = \
            data_sets[title]["original maximum value"]
    
    return preprocessing_function

def splitDataSet(data_dictionary, method = "default", fraction = 0.9):
    
    print("Splitting data set.")
    start_time = time()
    
    if method == "default":
        if self.split_indices:
            method = "indices"
        else:
            method = "random"
    
    method = normaliseString(method)
    
    M = data_dictionary["values"].shape[0]
    
    numpy.random.seed(42)
    
    if method == "random":
        
        M_training_validation = int(fraction * M)
        M_training = int(fraction * M_training_validation)
        
        shuffled_indices = numpy.random.permutation(M)
        
        training_indices = shuffled_indices[:M_training]
        validation_indices = shuffled_indices[M_training:M_training_validation]
        test_indices = shuffled_indices[M_training_validation:]
    
    elif method == "indices":
        
        split_indices = data_dictionary["split indices"]
        
        training_indices = split_indices["training"]
        test_indices = split_indices["test"]
        
        if "validation" in split_indices:
            validation_indices = split_indices["validation"]
        else:
            M_training_validation = training_indices.stop
            M_all = test_indices.stop
            
            M_training = M_training_validation - (M_all - M_training_validation)
            # M_training = int(fraction * M_training_validation)
            
            training_indices = slice(M_training)
            validation_indices = slice(M_training, M_training_validation)
    
    split_data_dictionary = {
        "training set": {
            "values": data_dictionary["values"][training_indices],
            "preprocessed values": None,
            "binarised values": None,
            "labels": None,
            "example names": data_dictionary["example names"][training_indices]
        },
        "validation set": {
            "values": data_dictionary["values"][validation_indices],
            "preprocessed values": None,
            "binarised values": None,
            "labels": None,
            "example names": data_dictionary["example names"][validation_indices]
        },
        "test set": {
            "values": data_dictionary["values"][test_indices],
            "preprocessed values": None,
            "binarised values": None,
            "labels": None,
            "example names": data_dictionary["example names"][test_indices]
        },
    }
    
    if "labels" in data_dictionary and data_dictionary["labels"] is not None:
        split_data_dictionary["training set"]["labels"] = \
            data_dictionary["labels"][training_indices]
        split_data_dictionary["validation set"]["labels"] = \
            data_dictionary["labels"][validation_indices]
        split_data_dictionary["test set"]["labels"] = \
            data_dictionary["labels"][test_indices]
    
    if "preprocessed values" in data_dictionary \
        and data_dictionary["preprocessed values"] is not None:
        
        split_data_dictionary["training set"]["preprocessed values"] = \
            data_dictionary["preprocessed values"][training_indices]
        split_data_dictionary["validation set"]["preprocessed values"] = \
            data_dictionary["preprocessed values"][validation_indices]
        split_data_dictionary["test set"]["preprocessed values"] = \
            data_dictionary["preprocessed values"][test_indices]
    
    if "binarised values" in data_dictionary \
        and data_dictionary["binarised values"] is not None:
        
        split_data_dictionary["training set"]["binarised values"] = \
            data_dictionary["binarised values"][training_indices]
        split_data_dictionary["validation set"]["binarised values"] = \
            data_dictionary["binarised values"][validation_indices]
        split_data_dictionary["test set"]["binarised values"] = \
            data_dictionary["binarised values"][test_indices]
    
    duration = time() - start_time
    print("Data set split ({}).".format(formatDuration(duration)))
    
    numpy.random.seed()
    
    return split_data_dictionary

def loadFromSparseData(path):
    
    start_time = time()
    
    def converter(data):
        if type(data) == scipy.sparse.csr.csr_matrix:
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
    
    duration = time() - start_time
    print("Data loaded from sparse representation" +
        " ({}).".format(formatDuration(duration)))
    
    return data_dictionary

def saveAsSparseData(data_dictionary, path):
    
    start_time = time()
    
    sparse_data_dictionary = {}
    
    def converter(data):
        if type(data) == numpy.ndarray and data.ndim == 2:
            return scipy.sparse.csr_matrix(data)
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
    
    duration = time() - start_time
    print("Data saved in sparse representation" +
        " ({}).".format(formatDuration(duration)))

def loadMouseRetinaDataSet(paths):
    
    values_data = pandas.read_csv(paths["values"]["full"], sep='\s+',
        index_col = 0, compression = "gzip", engine = "python"
    )
    
    values = values_data.values.T
    example_names = numpy.array(values_data.columns.tolist())
    feature_names = numpy.array(values_data.index.tolist())
    
    values = values.astype(float)
    
    labels = numpy.zeros(example_names.shape, int)
    
    with open(paths["labels"]["full"], "r") as labels_data:
        for line in labels_data.read().split("\n"):
            
            if line == "":
                continue
            
            example_name, label = line.split("\t")
            
            labels[example_names == example_name] = int(label)
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    return data_dictionary

def loadMNISTDataSet(paths):
    
    values = {}
    
    for kind in paths["values"]:
        with gzip.open(paths["values"][kind], "rb") as values_stream:
            _, M, r, c = struct.unpack(">IIII", values_stream.read(16))
            values_buffer = values_stream.read(M * r * c)
            values_flat = numpy.frombuffer(values_buffer, dtype = numpy.uint8)
            values[kind] = values_flat.reshape(-1, r * c)
    
    N = r * c
    
    labels = {}
    
    for kind in paths["labels"]:
        with gzip.open(paths["labels"][kind], "rb") as labels_stream:
            _, M = struct.unpack(">II", labels_stream.read(8))
            labels_buffer = labels_stream.read(M)
            labels[kind] = numpy.frombuffer(labels_buffer, dtype = numpy.int8)
    
    M_training = values["training"].shape[0]
    M_test = values["test"].shape[0]
    M = M_training + M_test
    
    split_indices = {
        "training": slice(0, M_training),
        "test": slice(M_training, M)
    }
    
    values = numpy.concatenate((values["training"], values["test"]))
    labels = numpy.concatenate((labels["training"], labels["test"]))
    
    values = values.astype(float)
    
    example_names = numpy.array(["image {}".format(i + 1) for i in range(M)])
    feature_names = numpy.array(["pixel {}".format(j + 1) for j in range(N)])
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }
    
    return data_dictionary

def loadNormalisedMNISTDataSet(paths):
    
    with gzip.open(paths["all"]["full"], "r") as data_file:
        (values_training, labels_training), (values_validation, \
            labels_validation), (values_test, labels_test) \
            = pickle.load(data_file, encoding = "latin1")
    
    M_training = values_training.shape[0]
    M_validation = values_validation.shape[0]
    M_training_validation = M_training + M_validation
    M_test = values_test.shape[0]
    M = M_training_validation + M_test
    
    split_indices = {
        "training": slice(0, M_training),
        "validation": slice(M_training, M_training_validation),
        "test": slice(M_training_validation, M)
    }
    
    values = numpy.concatenate((
        values_training, values_validation, values_test
    ))
    
    labels = numpy.concatenate((
        labels_training, labels_validation, labels_test
    ))
    
    N = values.shape[1]
    
    example_names = numpy.array(["image {}".format(i + 1) for i in range(M)])
    feature_names = numpy.array(["pixel {}".format(j + 1) for j in range(N)])
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }
    
    return data_dictionary

def loadBinarisedMNISTDataSet(paths):
    
    values = {}
    
    for kind in paths["values"]:
        values[kind] = numpy.loadtxt(paths["values"][kind])
    
    M_training = values["training"].shape[0]
    M_validation = values["validation"].shape[0]
    M_training_validation = M_training + M_validation
    M_test = values["test"].shape[0]
    M = M_training_validation + M_test
    
    split_indices = {
        "training": slice(0, M_training),
        "validation": slice(M_training, M_training_validation),
        "test": slice(M_training_validation, M)
    }
    
    values = numpy.concatenate((
        values["training"], values["validation"], values["test"]
    ))
    
    N = values.shape[1]
    
    example_names = numpy.array(["image {}".format(i + 1) for i in range(M)])
    feature_names = numpy.array(["pixel {}".format(j + 1) for j in range(N)])
    
    data_dictionary = {
        "values": values,
        "labels": None,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }
    
    return data_dictionary

def loadReutersDataSet(paths):
    
    topics_list = []
    body_list = []
    
    with tarfile.open(paths["all"]["full"], 'r:gz') as tarball:
        
        article_filenames = [f for f in tarball.getnames() if ".sgm" in f]
        
        for article_filename in article_filenames:
            
            with tarball.extractfile(article_filename) as article_html:
                soup = BeautifulSoup(article_html, 'html.parser')
            
            for article in soup.find_all("reuters"):
                
                topics = article.topics
                body = article.body
                
                if topics is not None and body is not None:
                    
                    topics_generator = topics.find_all("d")
                    topics_text = [topic.get_text() for topic in topics_generator]
                    
                    body_text = body.get_text()
                    
                    if len(topics_text) > 0 and len(body_text) > 0:
                        topics_list.append(topics_text)
                        body_list.append(body_text)
    
    M = len(body_list)
    
    bag_of_words, distinct_words = createBagOfWords(body_list)
    
    values = bag_of_words
    labels = numpy.array([t[0] for t in topics_list])
    example_names = numpy.array(["article {}".format(i + 1) for i in range(M)])
    feature_names = numpy.array(distinct_words)
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    return data_dictionary

def load20NewsgroupsDataSet(paths):
    
    documents = {
        "train": [],
        "test": []
    }
    document_ids = {
        "train": [],
        "test": []
    }
    newsgroups = {
        "train": [],
        "test": []
    }
    
    with tarfile.open(paths["all"]["full"], 'r:gz') as tarball:
        
        for member in tarball:
            if member.isfile():
                
                with tarball.extractfile(member) as document_file:
                    document = document_file.read().decode("latin1") 
                
                kind, newsgroup, document_id = member.name.split(os.sep)
                
                kind = kind.split("-")[-1]
                
                documents[kind].append(document)
                document_ids[kind].append(document_id)
                newsgroups[kind].append(newsgroup)
    
    M_train = len(documents["train"])
    M_test = len(documents["test"])
    M = M_train + M_test
    
    split_indices = {
        "training": slice(0, M_train),
        "test": slice(M_train, M)
    }
    
    documents = documents["train"] + documents["test"]
    document_ids = document_ids["train"] + document_ids["test"]
    newsgroups = newsgroups["train"] + newsgroups["test"]
    
    bag_of_words, distinct_words = createBagOfWords(documents)
    
    values = bag_of_words
    labels = numpy.array(newsgroups)
    example_names = numpy.array(document_ids)
    feature_names = numpy.array(distinct_words)
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }
    
    return data_dictionary

def loadSampleDataSet(paths):
    
    with gzip.open(paths["all"]["full"], "rb") as data_file:
        data = pickle.load(data_file)
    
    values = data["values"]
    labels = data["labels"]
    
    M, N = values.shape
    
    example_names = numpy.array(["example {}".format(i + 1) for i in range(M)])
    feature_names = numpy.array(["feature {}".format(j + 1) for j in range(N)])
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    return data_dictionary

def loadDevelopmentDataSet(number_of_examples = 10000, number_of_features = 25,
    scale = 10, update_probability = 0.0001):
    
    numpy.random.seed(60)
        
    M = number_of_examples
    N = number_of_features
    
    values = numpy.empty((M, N))
    labels = numpy.empty(M, int)
    
    r = numpy.empty((M, N))
    p = numpy.empty((M, N))
    dropout = numpy.empty((M, N))
    
    r_draw = lambda: scale * numpy.random.rand(N)
    p_draw = lambda: numpy.random.rand(N)
    dropout_draw = lambda: numpy.random.rand(N)
    
    r_type = r_draw()
    p_type = p_draw()
    dropout_type = dropout_draw()
    
    label = 1
    
    for i in range(M):
        u = numpy.random.rand()
        if u > 1 - update_probability:
            r_type = r_draw()
            p_type = p_draw()
            dropout_type = dropout_draw()
            label += 1
        r[i] = r_type
        p[i] = p_type
        dropout[i] = dropout_type
        labels[i] = label
    
    shuffled_indices = numpy.random.permutation(M)

    r = r[shuffled_indices]
    p = p[shuffled_indices]
    dropout = dropout[shuffled_indices]
    labels = labels[shuffled_indices]
    
    no_class_indices = numpy.random.permutation(M)[:int(0.1 * M)]
    labels[no_class_indices] = 0
    
    for i in range(M):
        for j in range(N):
            # value = numpy.random.poisson(r[i, j])
            value = numpy.random.negative_binomial(r[i, j], p[i, j])
            value_dropout = numpy.random.binomial(1, dropout[i, j])
            # value_dropout = numpy.random.poisson(dropout[i, j])
            values[i, j] = value_dropout * value
    
    example_names = numpy.array(["example {}".format(i + 1) for i in range(M)])
    feature_names = numpy.array(["feature {}".format(j + 1) for j in range(N)])
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    numpy.random.seed()
        
    return data_dictionary

def createBagOfWords(documents):
    
    def findWords(text):
        lower_case_text = text.lower()
        # lower_case_text = re.sub(r"(reuter)$", "", lower_case_text)
        lower_case_text = re.sub(r"\d+[\d.,\-\(\)+]*", " DIGIT ", lower_case_text)
        words = re.compile(r"[\w'\-]+").findall(lower_case_text)
        words = [stemming.stem(word) for word in words]
        return words
    
    # Create original bag of words with one bucket per distinct word. 
    
    # List and set for saving the found words
    documents_words = list()
    distinct_words = set()
    
    # Run through documents bodies and update the list and set with words from findWords()
    for document in documents:
        
        words = findWords(document)
        
        documents_words.append(words)
        distinct_words.update(words)
    
    # Create list of the unique set of distinct words found
    distinct_words = list(distinct_words)
    
    # Create dictionary mapping words to their index in the list
    distinct_words_index = dict()
    for i, distinct_word in enumerate(distinct_words):
        distinct_words_index[distinct_word] = i
    
    # Initialize bag of words matrix with numpy's zeros()
    bag_of_words = numpy.zeros([len(documents), len(distinct_words)])
    
    # Fill out bag of words with cumulative count of word occurences
    for i, words in enumerate(documents_words):
        for word in words:
            bag_of_words[i, distinct_words_index[word]] += 1
    
    # Return bag of words matrix as a sparse representation matrix to save memory
    return bag_of_words, distinct_words

## Apply weights
def applyWeights(data, method, preprocessPath = None):
    
    weights = loadWeights(data, method, preprocessPath)
    
    return weights * data

def loadWeights(data, method, preprocessPath):
    
    if preprocessPath:
        weights_path = preprocessPath(method + "-weights")
    else:
        weights_path = None
    
    if weights_path and os.path.isfile(weights_path):
        print("Loading weights from sparse representation.")
        weights_dictionary = loadFromSparseData(weights_path)
    else:
        if method == "gini":
            weights = computeGiniIndices(data)
        elif method == "idf":
            weights = computeInverseGlobalFrequencyWeights(data)
        
        weights_dictionary = {"weights": weights}
        
        if weights_path:
            print("Saving weights in sparse representation.")
            saveAsSparseData(weights_dictionary, weights_path)
    
    return weights_dictionary["weights"]

## Compute Gini indices
def computeGiniIndices(data, epsilon = 1e-16, batch_size = 5000):
    """Calculate the Gini coefficients along last axis of a NumPy array."""
    # Based on last equation on:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    
    print("Computing Gini indices.")
    start_time = time()
    
    # Number of examples, M, and features, N
    M, N = data.shape
    
    # 1-indexing vector for each data element
    index_vector = 2 * numpy.arange(1, M + 1) - M - 1
    
    # Values cannot be 0
    data = numpy.clip(data, epsilon, data)
    
    gini_indices = numpy.zeros(N)
    
    for i in range(0, N, batch_size):
        batch = data[:, i:(i+batch_size)]
        
        # Array should be normalized and sorted frequencies over the examples
        batch = numpy.sort(batch / (numpy.sum(batch, axis = 0)), axis = 0)
        
        #Gini coefficients over the examples for each feature. 
        gini_indices[i:(i+batch_size)] = index_vector @ batch / M
    
    duration = time() - start_time
    print("Gini indices computed ({}).".format(formatDuration(duration)))
    
    return gini_indices

def computeInverseGlobalFrequencyWeights(data):
    
    print("Computing IDF weights.")
    start_time = time()
    
    M = data.shape[0]
    
    global_frequencies = numpy.sum(numpy.where(data > 0, 1, 0), axis=0)
    
    idf_weights = numpy.log(M / (global_frequencies + 1))
    
    duration = time() - start_time
    print("IDF weights computed ({}).".format(formatDuration(duration)))
    
    return idf_weights

def supersetLabels(labels, label_superset):
    
    if not label_superset:
        return None
    
    label_superset_reverse = {v: k for k, vs in label_superset.items()
        for v in vs}
    
    label_to_superset_label = lambda label: label_superset_reverse[label]
    labels_to_superset_labels = numpy.vectorize(label_to_superset_label)
    
    superset_labels = labels_to_superset_labels(labels)
    
    return superset_labels

def supersetClassPalette(class_palette, label_superset):
    
    if not label_superset:
        return None
    
    superset_class_palette = {}
    
    for superset_label, labels_in_superset_label in label_superset.items():
        superset_label_colours = []
        for label_in_superset_label in labels_in_superset_label:
            superset_label_colours.append(
                class_palette[label_in_superset_label]
            )
        superset_class_palette[superset_label] = \
            numpy.array(superset_label_colours).mean(axis = 0).tolist()
    
    return superset_class_palette

def directory(base_directory, data_set, splitting_method, splitting_fraction,
    preprocessing = True):
    
    data_set_directory = os.path.join(base_directory, data_set.name)
    
    if splitting_method == "default":
        splitting_method = data_set.defaultSplittingMethod()
    
    splitting_directory = "split-{}-{}".format(splitting_method,
        splitting_fraction)
    
    preprocessing_directory_parts = []
    
    if data_set.feature_selection:
        preprocessing_directory_parts.append(normaliseString(
            data_set.feature_selection))
        if data_set.feature_parameter:
            preprocessing_directory_parts.append(str(
                data_set.feature_parameter))
    
    if data_set.example_filter:
        preprocessing_directory_parts.append(normaliseString(
            data_set.example_filter))
        if data_set.example_filter_parameters:
            preprocessing_directory_parts.extend(map(normaliseString,
                data_set.example_filter_parameters))
    
    if preprocessing and data_set.preprocessing_methods:
        preprocessing_directory_parts.extend(map(normaliseString,
            data_set.preprocessing_methods))
    
    if preprocessing and data_set.noisy_preprocessing_methods:
        preprocessing_directory_parts.append("noisy")
        preprocessing_directory_parts.extend(map(normaliseString,
            data_set.noisy_preprocessing_methods))
    
    if preprocessing_directory_parts:
        preprocessing_directory = "-".join(preprocessing_directory_parts)
    else:
        preprocessing_directory = "none"
    
    directory = os.path.join(data_set_directory, splitting_directory,
        preprocessing_directory)
    
    return directory
