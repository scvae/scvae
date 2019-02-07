# ======================================================================== #
# 
# Copyright (c) 2017 - 2018 scVAE authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# ======================================================================== #

import os
import gzip
import tarfile
import pickle
import struct
import random

import re
from bs4 import BeautifulSoup

import pandas
import tables
import json

import numpy
import scipy.sparse
import scipy.io
import sklearn.preprocessing
import stemming.porter2 as stemming

from functools import reduce

import seaborn

from time import time

from auxiliary import (
    formatDuration,
    normaliseString, properString, enumerateListOfStrings,
    isfloat,
    downloadFile, copyFile
)

from miscellaneous.decomposition import (
    decompose,
    DECOMPOSITION_METHOD_NAMES,
    DECOMPOSITION_METHOD_LABEL
)

preprocess_suffix = "preprocessed"
original_suffix = "original"
preprocessed_extension = ".sparse.h5"

maximum_duration_before_saving = 30 # seconds

subset_kinds = ["full", "training", "validation", "test"]

data_sets = {
    "Macosko-MRC": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "values": {
                "full": "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63472/suppl/GSE63472_P14Retina_merged_digital_expression.txt.gz"
            },
            "labels": {
                "full": "http://mccarrolllab.com/wp-content/uploads/2015/05/retina_clusteridentities.txt"
            }
        },
        "loading function": lambda x: loadMouseRetinaDataSet(x),
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
            "Müller glia": [34],
            "Others": [i for i in range(35, 40)],
            "No class": [0]
        },
        "sorted superset class names": [
            "Horizontal",
            "Retinal ganglion",
            "Amacrine",
            "Rods",
            "Cones",
            "Bipolar",
            "Müller glia"
        ],
        "literature probabilities": {
            "Horizontal": 0.5 / 100,
            "Retinal ganglion": 0.5 / 100,
            "Amacrine": 0.07,
            "Rods": 79.9 / 100,
            "Cones": 2.1 / 100,
            "Bipolar": 7.3 / 100,
            "Müller glia": 2.8 / 100,
            "Others": 0
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
    
    "10x-MBC-20k": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "values": {
                "full": "http://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_neuron20k.h5"
            },
            "labels": {
                "full": None
            }
        },
        "loading function": lambda x: load10xDataSet(x),
        "example type": "counts"
    },
    
    "10x-MBC": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "values": {
                "full": "http://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5"
            },
            "labels": {
                "full": None
            }
        },
        "loading function": lambda x: load10xDataSet(x),
        "example type": "counts"
    },
    
    "10x-PBMC-PL": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "all": {
                "CD56+ natural killer cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz",
                "CD19+ B cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz",
                "CD4+/CD25+ regulatory T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz"
            },
        },
        "loading function": lambda x: loadAndCombine10xDataSets(x),
        "example type": "counts"
    },
    
    "10x-PBMC-PT": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "all": {
                "CD8+/CD45RA+ naïve cytotoxic T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz",
                "CD4+/CD25+ regulatory T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz",
                "CD4+/CD45RA+/CD25- naïve T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_t/naive_t_filtered_gene_bc_matrices.tar.gz"
            },
        },
        "loading function": lambda x: loadAndCombine10xDataSets(x),
        "example type": "counts"
    },
    
    "10x-PBMC-PP": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "all": {
                "CD19+ B cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz",
                "CD34+ cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd34/cd34_filtered_gene_bc_matrices.tar.gz",
                "CD4+ helper T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd4_t_helper/cd4_t_helper_filtered_gene_bc_matrices.tar.gz",
                "CD4+/CD25+ regulatory T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/regulatory_t/regulatory_t_filtered_gene_bc_matrices.tar.gz",
                "CD4+/CD45RA+/CD25- naïve T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_t/naive_t_filtered_gene_bc_matrices.tar.gz",
                "CD4+/CD45RO+ memory T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/memory_t/memory_t_filtered_gene_bc_matrices.tar.gz",
                "CD56+ natural killer cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cd56_nk/cd56_nk_filtered_gene_bc_matrices.tar.gz",
                "CD8+ cytotoxic T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/cytotoxic_t/cytotoxic_t_filtered_gene_bc_matrices.tar.gz",
                "CD8+/CD45RA+ naïve cytotoxic T cells": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/naive_cytotoxic/naive_cytotoxic_filtered_gene_bc_matrices.tar.gz"
            },
        },
        "loading function": lambda x: loadAndCombine10xDataSets(x),
        "example type": "counts"
    },
    
    "10x-PBMC-68k": {
        "tags": {
            "example": "cell",
            "feature": "gene",
            "class": "cell type",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "values": {
                "full": "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz"
            },
            "labels": {
                "full": "https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv"
            }
        },
        "loading function": lambda x: load10xDataSet(x),
        "example type": "counts"
    },
    
    "TCGA-Kallisto": {
        "tags": {
            "example": "sample",
            "feature": "gene ID",
            "class": "tissue site",
            "mapped feature": "gene",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "values": {
                "full": "https://toil.xenahubs.net/download/tcga_Kallisto_est_counts.gz"
            },
            "labels": {
                "full": "https://tcga.xenahubs.net/download/TCGA.PANCAN.sampleMap/PANCAN_clinicalMatrix.gz"
            },
            "feature mapping": {
                "full": "https://toil.xenahubs.net/download/gencode.v23.annotation.transcript.probemap.gz"
            }
        },
        "loading function": lambda x: loadTCGADataSet(x),
        "example type": "counts"
    },
    
    "TCGA-RSEM": {
        "tags": {
            "example": "sample",
            "feature": "gene ID",
            "mapped feature": "gene",
            "class": "tissue site",
            "type": "count",
            "item": "transcript"
        },
        "URLs": {
            "values": {
                "full": "https://toil.xenahubs.net/download/tcga_gene_expected_count.gz"
            },
            "labels": {
                "full": "https://tcga.xenahubs.net/download/TCGA.PANCAN.sampleMap/PANCAN_clinicalMatrix.gz"
            },
            "feature mapping": {
                "full": "https://toil.xenahubs.net/download/gencode.v23.annotation.gene.probeMap.gz"
            }
        },
        "loading function": lambda x: loadTCGADataSet(x),
        "example type": "counts"
    },
    
    "MNIST (original)": {
        "tags": {
            "example": "digit",
            "feature": "pixel",
            "class": "value",
            "type": "count",
            "item": "intensity"
        },
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
        "loading function": lambda x: loadMNISTDataSet(x),
        "maximum value": 255,
        "example type": "images",
        "feature dimensions": (28, 28)
    },
    
    "MNIST (normalised)": {
        "tags": {
            "example": "digit",
            "feature": "pixel",
            "class": "value",
            "type": "value",
            "item": "intensity"
        },
        "URLs": {
            "all": {
                    "full": "http://deeplearning.net/data/mnist/mnist.pkl.gz"
            }
        },
        "loading function": lambda x: loadNormalisedMNISTDataSet(x),
        "maximum value": 1,
        "example type": "images",
        "feature dimensions": (28, 28)
    },
    
    "MNIST (binarised)": {
        "tags": {
            "example": "digit",
            "feature": "pixel",
            "class": "value",
            "type": "value",
            "item": "intensity"
        },
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
        "loading function": lambda x: loadBinarisedMNISTDataSet(x),
        "maximum value": 1,
        "example type": "images",
        "feature dimensions": (28, 28)
    },
    
    "Reuters": {
        "tags": {
            "example": "document",
            "feature": "word",
            "class": "topic",
            "type": "count",
            "item": "word"
        },
        "URLs": {
            "all": {
                "full": "http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz"
            }
        },
        "loading function": lambda x: loadReutersDataSet(x)
    },
    
    "20 Newsgroups": {
        "tags": {
            "example": "document",
            "feature": "word",
            "class": "topic",
            "type": "count",
            "item": "word"
        },
        "URLs": {
            "all": {
                "full":
                    "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
            }
        },
        "loading function": lambda x: load20NewsgroupsDataSet(x),
        "example type": "counts"
    },
    
    "blobs": {
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/maxvo/datasets/blobs.pkl.gz"
            }
        },
        "loading function": lambda x: loadSampleDataSet(x),
        "example type": "dummy"
    },
    
    "circles": {
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/maxvo/datasets/circles.pkl.gz"
            }
        },
        "loading function": lambda x: loadSampleDataSet(x),
        "example type": "dummy"
    },
    
    "moons": {
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/maxvo/datasets/moons.pkl.gz"
            }
        },
        "loading function": lambda x: loadSampleDataSet(x),
        "example type": "dummy"
    },
    
    "sample": {
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/chegr/data-sets/count_samples.pkl.gz"
            }
        },
        "loading function": lambda x: loadSampleDataSet(x),
        "example type": "counts"
    },
    
    "sample (sparse)": {
        "URLs": {
            "all": {
                "full": "http://people.compute.dtu.dk/chegr/data-sets/count_samples_sparse.pkl.gz"
            }
        },
        "loading function": lambda x: loadSampleDataSet(x),
        "example type": "counts"
    },
    
    "development": {
        "URLs": {},
        "loading function": lambda x: loadDevelopmentDataSet(
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
            "Rods": [1],
            "Cones": [2, 3],
            "No class": [0]
        },
        "sorted superset class names": [
            "Rods",
            "Cones"
        ],
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
    def __init__(self, input_file_or_name,
        values = None,
        total_standard_deviations = None,
        explained_standard_deviations = None,
        preprocessed_values = None, binarised_values = None,
        labels = None, class_names = None,
        example_names = None, feature_names = None,
        map_features = False, features_mapped = False,
        feature_selection = [], example_filter = [],
        preprocessing_methods = [], preprocessed = None,
        binarise_values = False,
        noisy_preprocessing_methods = [],
        kind = "full", version = "original",
        directory = "data"):
        
        super(DataSet, self).__init__()
        
        # Name of data set and optional entry for data sets dictionary
        self.name, data_set_dictionary = parseInput(input_file_or_name)
        
        # Directories and paths for data set
        self.directory = os.path.join(directory, self.name)
        self.preprocess_directory = os.path.join(self.directory,
            preprocess_suffix)
        self.original_directory = os.path.join(self.directory,
            original_suffix)
        self.preprocessedPath = preprocessedPathFunction(
            self.preprocess_directory, self.name)
        
        # Save data set dictionary if necessary
        if data_set_dictionary:
            saveDataSetDictionaryAsJSONFile(data_set_dictionary,
                self. directory)
        
        # Find data set
        self.title = findDataSet(self.name, directory)
        
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
        
        # Class mapper for data set
        self.class_mapper = dataSetClassMapper(self.title)
        
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
        
        # Predicted labels
        
        self.predicted_cluster_ids = None
        
        self.predicted_labels = None
        self.predicted_class_names = None
        self.number_of_predicted_classes = None
        self.predicted_class_palette = None
        self.predicted_label_sorter = None
        
        self.predicted_superset_labels = None
        self.predicted_superset_class_names = None
        self.number_of_predicted_superset_classes = None
        self.predicted_superset_class_palette = None
        self.predicted_superset_label_sorter = None
        
        # Sorted class names for data set
        sorted_class_names = dataSetSortedClassNames(self.title)
        self.label_sorter = createLabelSorter(sorted_class_names)
        sorted_superset_class_names = dataSetSortedClassNames(self.title,
            superset = True)
        self.superset_label_sorter = createLabelSorter(
            sorted_superset_class_names)
        
        # Feature mapping
        self.map_features = map_features
        self.feature_mapping = None
        self.features_mapped = features_mapped
        
        if self.features_mapped:
            self.tags = updateTagForMappedFeatures(self.tags)
        
        # Feature selection
        if feature_selection:
            self.feature_selection = feature_selection[0]
            if len(feature_selection) > 1:
                self.feature_selection_parameters = feature_selection[1:]
            else:
                self.feature_selection_parameters = None
        else:
            self.feature_selection = None
            self.feature_selection_parameters = None
        
        # Example filterering
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
        self.binarise_values = binarise_values
        
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
        
        if self.kind == "full" and self.values is None:
            
            print("Data set:")
            print("    title:", self.title)
            
            if self.map_features:
                print("    feature mapping: if available")
            
            if self.feature_selection:
                print("    feature selection:", self.feature_selection)
                if self.feature_selection_parameters:
                    print("        parameters:",
                        ", ".join(self.feature_selection_parameters))
                else:
                    print("        parameters: default")
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
    
    @property
    def has_values(self):
        return self.values is not None
    
    @property
    def has_preprocessed_values(self):
        return self.preprocessed_values is not None
    
    @property
    def has_binarised_values(self):
        return self.binarised_values is not None
    
    @property
    def has_labels(self):
        return self.labels is not None
    
    @property
    def has_superset_labels(self):
        return self.superset_labels is not None
    
    @property
    def has_predictions(self):
        return self.has_predicted_labels or self.has_predicted_cluster_ids
    
    @property
    def has_predicted_labels(self):
        return self.predicted_labels is not None
    
    @property
    def has_predicted_superset_labels(self):
        return self.predicted_superset_labels is not None
    
    @property
    def has_predicted_cluster_ids(self):
        return self.predicted_cluster_ids is not None
    
    def update(self, values = None,
        total_standard_deviations = None,
        explained_standard_deviations = None,
        preprocessed_values = None,
        binarised_values = None, labels = None,
        example_names = None, feature_names = None, class_names = None):
        
        if values is not None:
            
            self.values = values
            
            self.count_sum = self.values.sum(axis = 1).reshape(-1, 1)
            if isinstance(self.count_sum, numpy.matrix):
                self.count_sum = self.count_sum.A
            self.normalised_count_sum = self.count_sum / self.count_sum.max()
            
            M_values, N_values = values.shape
            
            if example_names is not None:
                self.example_names = example_names
                assert len(self.example_names.shape) == 1, \
                    "The list of example names is multi-dimensional: {}."\
                        .format(self.example_names.shape)
                M_examples = self.example_names.shape[0]
                assert M_values == M_examples, \
                    "The number of examples in the value matrix ({}) "\
                        .format(M_values) + \
                    "is not the same as the number of example names ({})."\
                        .format(M_examples)
            
            if feature_names is not None:
                self.feature_names = feature_names
                assert len(self.feature_names.shape) == 1, \
                    "The list of feature names is multi-dimensional: {}."\
                        .format(self.feature_names.shape)
                N_features = self.feature_names.shape[0]
                assert N_values == N_features, \
                    "The number of features in the value matrix ({}) "\
                        .format(N_values) + \
                    "is not the same as the number of feature names ({})."\
                        .format(N_features)
            
            self.number_of_examples = M_values
            self.number_of_features = N_values
        
        else:
            
            if example_names is not None and feature_names is not None:
                
                self.example_names = example_names
                self.feature_names = feature_names
        
        if labels is not None:
            
            if self.class_mapper:
                labels = labels.tolist()
                labels = [
                    properString(label, self.class_mapper, normalise = False)
                    for label in labels
                ]
                labels = numpy.array(labels)
            
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
            
            if not self.excluded_classes:
                for excluded_class in default_excluded_classes:
                    if excluded_class in self.class_names:
                        self.excluded_classes.append(excluded_class)
            
            self.number_of_classes = len(self.class_names)
            
            self.number_of_excluded_classes = len(self.excluded_classes) \
                if self.excluded_classes \
                else 0
            
            if self.label_superset:
                
                self.superset_labels = supersetLabels(
                    self.labels, self.label_superset)
                
                self.superset_class_names = numpy.unique(
                    self.superset_labels).tolist()
                
                self.superset_class_id_to_superset_class_name = {}
                self.superset_class_name_to_superset_class_id = {}
            
                for i, class_name in \
                    enumerate(self.superset_class_names):
                    self.superset_class_name_to_superset_class_id[class_name] = i
                    self.superset_class_id_to_superset_class_name[i] = class_name
                
                if not self.excluded_superset_classes:
                    for excluded_class in default_excluded_classes:
                        if excluded_class in self.superset_class_names:
                            self.excluded_superset_classes.append(excluded_class)
                
                self.number_of_superset_classes = len(self.superset_class_names)
                
                self.number_of_excluded_superset_classes = \
                    len(self.excluded_superset_classes) \
                    if self.excluded_superset_classes \
                    else 0
        
        if total_standard_deviations is not None:
            self.total_standard_deviations = total_standard_deviations
        
        if explained_standard_deviations is not None:
            self.explained_standard_deviations = explained_standard_deviations
        
        if preprocessed_values is not None:
            self.preprocessed_values = preprocessed_values
        
        if binarised_values is not None:
            self.binarised_values = binarised_values
    
    def updatePredictions(self, predicted_cluster_ids = None,
        predicted_labels = None, predicted_class_names = None,
        predicted_superset_labels = None,
        predicted_superset_class_names = None):
        
        if predicted_cluster_ids is not None:
            self.predicted_cluster_ids = predicted_cluster_ids
        
        if predicted_labels is not None:
            
            self.predicted_labels = predicted_labels
            
            if predicted_class_names is not None:
                self.predicted_class_names = predicted_class_names
            else:
                self.predicted_class_names = \
                    numpy.unique(self.predicted_labels).tolist()
            
            self.number_of_predicted_classes = len(self.predicted_class_names)
            
            if set(self.predicted_class_names) < set(self.class_names):
                self.predicted_class_palette = self.class_palette
                self.predicted_label_sorter = self.label_sorter
        
        if predicted_superset_labels is not None:
            
            self.predicted_superset_labels = predicted_superset_labels
            
            if predicted_superset_class_names is not None:
                self.predicted_superset_class_names = \
                    predicted_superset_class_names
            else:
                self.predicted_superset_class_names = \
                    numpy.unique(self.predicted_superset_labels).tolist()
            
            self.number_of_predicted_superset_classes = \
                len(self.predicted_superset_class_names)
            
            if set(self.predicted_superset_class_names) < \
                set(self.superset_class_names):
                
                self.predicted_superset_class_palette = \
                    self.superset_class_palette
                self.predicted_superset_label_sorter = \
                    self.superset_label_sorter
    
    def resetPredictions(self):
        
        self.predicted_cluster_ids = None
        
        self.predicted_labels = None
        self.predicted_class_names = None
        self.number_of_predicted_classes = None
        self.predicted_class_palette = None
        self.predicted_label_sorter = None
        
        self.predicted_superset_labels = None
        self.predicted_superset_class_names = None
        self.number_of_predicted_superset_classes = None
        self.predicted_superset_class_palette = None
        self.predicted_superset_label_sorter = None
    
    def load(self):
        
        sparse_path = self.preprocessedPath()
        
        if os.path.isfile(sparse_path):
            print("Loading data set.")
            data_dictionary = loadDataDictionary(sparse_path)
            print()
        else:
            original_paths = acquireDataSet(self.title,
                self.original_directory)
            
            loading_time_start = time()
            data_dictionary = loadOriginalDataSet(self.title,
                original_paths)
            loading_duration = time() - loading_time_start
            
            print()
            
            if loading_duration > maximum_duration_before_saving:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving data set.")
                saveDataDictionary(data_dictionary, sparse_path)
                
                print()
        
        data_dictionary["values"] = SparseRowMatrix(data_dictionary["values"])
    
        self.update(
            values = data_dictionary["values"],
            labels = data_dictionary["labels"],
            example_names = data_dictionary["example names"],
            feature_names = data_dictionary["feature names"]
        )
        
        if "split indices" in data_dictionary:
            self.split_indices = data_dictionary["split indices"]
        
        if "feature mapping" in data_dictionary:
            self.feature_mapping = data_dictionary["feature mapping"]
        else:
            self.map_features = False
        
        if not self.feature_selection_parameters:
            self.feature_selection_parameters = defaultFeatureParameters(
                self.feature_selection, self.number_of_features)
        
        self.preprocess()
        
        if self.binarise_values:
            self.binarise()
    
    def preprocess(self):
        
        if not self.map_features and not self.preprocessing_methods \
            and not self.feature_selection and not self.example_filter:
            self.update(preprocessed_values = None)
            return
        
        sparse_path = self.preprocessedPath(
            map_features = self.map_features,
            preprocessing_methods = self.preprocessing_methods,
            feature_selection = self.feature_selection, 
            feature_selection_parameters = self.feature_selection_parameters,
            example_filter = self.example_filter,
            example_filter_parameters = self.example_filter_parameters
        )
        
        if os.path.isfile(sparse_path):
            print("Loading preprocessed data.")
            data_dictionary = loadDataDictionary(sparse_path)
            if "preprocessed values" not in data_dictionary:
                data_dictionary["preprocessed values"] = None
            if self.map_features:
                self.features_mapped = True
                self.tags = updateTagForMappedFeatures(self.tags)
            print()
        else:
            
            preprocessing_time_start = time()
            
            values = self.values
            example_names = self.example_names
            feature_names = self.feature_names
            
            if self.map_features and not self.features_mapped:
                
                print("Mapping {} original features to {} new features."
                    .format(self.number_of_features, len(self.feature_mapping))
                )
                start_time = time()
                
                values, feature_names = mapFeatures(
                    values, feature_names, self.feature_mapping)
                
                self.features_mapped = True
                self.tags = updateTagForMappedFeatures(self.tags)
                
                duration = time() - start_time
                print("Features mapped ({}).".format(formatDuration(duration)))
                
                print()
            
            if not self.preprocessed and self.preprocessing_methods:
                
                print("Preprocessing values.")
                start_time = time()
                
                preprocessing_function = preprocessingFunctionForDataSet(
                    self.title,
                    self.preprocessing_methods,
                    self.preprocessedPath
                )
                preprocessed_values = preprocessing_function(values)
                
                duration = time() - start_time
                print("Values preprocessed ({}).".format(formatDuration(duration)))
                
                print()
            
            else:
                preprocessed_values = None
            
            if self.feature_selection:
                values_dictionary, feature_names = selectFeatures(
                    {"original": values,
                     "preprocessed": preprocessed_values},
                    self.feature_names,
                    self.feature_selection,
                    self.feature_selection_parameters,
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
            
            if self.features_mapped or self.feature_selection:
                data_dictionary["feature names"] = feature_names
            
            if self.example_filter:
                data_dictionary["example names"] = example_names
                data_dictionary["labels"] = labels
            
            preprocessing_duration = time() - preprocessing_time_start
            
            if preprocessing_duration > maximum_duration_before_saving:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
            
                print("Saving preprocessed data set.")
                saveDataDictionary(data_dictionary, sparse_path)
                print()
        
        values = data_dictionary["values"]
        preprocessed_values = data_dictionary["preprocessed values"]
        
        if preprocessed_values is None:
            preprocessed_values = values
        
        if self.features_mapped or self.feature_selection:
            feature_names = data_dictionary["feature names"]
        else:
            feature_names = self.feature_names
        
        if self.example_filter:
            example_names = data_dictionary["example names"]
            labels = data_dictionary["labels"]
        else:
            example_names = self.example_names
            labels = self.labels
        
        values = SparseRowMatrix(values)
        preprocessed_values = SparseRowMatrix(preprocessed_values)
        
        self.update(
            values = values,
            preprocessed_values = preprocessed_values,
            feature_names = feature_names,
            example_names = example_names,
            labels = labels
        )
    
    def binarise(self):
        
        if self.preprocessed_values is None:
            raise NotImplementedError("Data set values have to have been",
                "preprocessed and feature selected first.")
        
        binarise_preprocessing = ["binarise"]
        
        sparse_path = self.preprocessedPath(
            map_features = self.map_features,
            preprocessing_methods = binarise_preprocessing,
            feature_selection = self.feature_selection, 
            feature_selection_parameters = self.feature_selection_parameters,
            example_filter = self.example_filter,
            example_filter_parameters = self.example_filter_parameters
        )
        
        if os.path.isfile(sparse_path):
            print("Loading binarised data.")
            data_dictionary = loadDataDictionary(sparse_path)
        
        else:
            
            binarising_time_start = time()
            
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
            
            binarising_duration = time() - binarising_time_start
            
            if binarising_duration > maximum_duration_before_saving:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving binarised data set.")
                saveDataDictionary(data_dictionary, sparse_path)
        
        binarised_values = SparseRowMatrix(binarised_values)
        
        self.update(
            binarised_values = data_dictionary["preprocessed values"],
        )
    
    def defaultSplittingMethod(self):
        if self.split_indices:
            return "indices"
        else:
            return "random"
    
    def split(self, method = "default", fraction = 0.9):
        
        if method == "default":
            method = self.defaultSplittingMethod()
        
        sparse_path = self.preprocessedPath(
            map_features = self.map_features,
            preprocessing_methods = self.preprocessing_methods,
            feature_selection = self.feature_selection,
            feature_selection_parameters = self.feature_selection_parameters,
            example_filter = self.example_filter,
            example_filter_parameters = self.example_filter_parameters,
            splitting_method = method,
            splitting_fraction = fraction,
            split_indices = self.split_indices
        )
        
        print("Splitting:")
        print("    method:", method)
        if method != "indices":
            print("    fraction: {:.1f} %".format(100 * fraction))
        print()
        
        if os.path.isfile(sparse_path):
            print("Loading split data sets.")
            split_data_dictionary = loadDataDictionary(sparse_path)
            if self.map_features:
                self.features_mapped = True
                self.tags = updateTagForMappedFeatures(self.tags)
            print()
        else:
            
            if self.values is None:
                self.load()
            
            data_dictionary = {
                "values": self.values,
                "preprocessed values": self.preprocessed_values,
                "binarised values": self.binarised_values,
                "labels": self.labels,
                "example names": self.example_names,
                "feature names": self.feature_names,
                "class names": self.class_names,
                "split indices": self.split_indices
            }
            
            splitting_time_start = time()
            split_data_dictionary = splitDataSet(data_dictionary, method,
                fraction)
            splitting_duration = time() - splitting_time_start
            
            print()
            
            if splitting_duration > maximum_duration_before_saving:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving split data sets.")
                saveDataDictionary(split_data_dictionary, sparse_path)
                print()
        
        for data_subset in split_data_dictionary:
            if not isinstance(split_data_dictionary[data_subset], dict):
                continue
            for data_subset_key in split_data_dictionary[data_subset]:
                if "values" in data_subset_key:
                    values = split_data_dictionary[data_subset][data_subset_key]
                    if values is not None:
                        split_data_dictionary[data_subset][data_subset_key] \
                            = SparseRowMatrix(values)
        
        training_set = DataSet(
            self.name,
            values = split_data_dictionary["training set"]["values"],
            preprocessed_values = \
                split_data_dictionary["training set"]["preprocessed values"],
            binarised_values = \
                split_data_dictionary["training set"]["binarised values"],
            labels = split_data_dictionary["training set"]["labels"],
            example_names = split_data_dictionary["training set"]["example names"],
            feature_names = split_data_dictionary["feature names"],
            features_mapped = self.features_mapped,
            class_names = split_data_dictionary["class names"],
            feature_selection = self.feature_selection,
            example_filter = self.example_filter,
            preprocessing_methods = self.preprocessing_methods,
            noisy_preprocessing_methods = self.noisy_preprocessing_methods,
            kind = "training"
        )
        
        validation_set = DataSet(
            self.name,
            values = split_data_dictionary["validation set"]["values"],
            preprocessed_values = \
                split_data_dictionary["validation set"]["preprocessed values"],
            binarised_values = \
                split_data_dictionary["validation set"]["binarised values"],
            labels = split_data_dictionary["validation set"]["labels"],
            example_names = split_data_dictionary["validation set"]["example names"],
            feature_names = split_data_dictionary["feature names"],
            features_mapped = self.features_mapped,
            class_names = split_data_dictionary["class names"],
            feature_selection = self.feature_selection,
            example_filter = self.example_filter,
            preprocessing_methods = self.preprocessing_methods,
            noisy_preprocessing_methods = self.noisy_preprocessing_methods,
            kind = "validation"
        )
        
        test_set = DataSet(
            self.name,
            values = split_data_dictionary["test set"]["values"],
            preprocessed_values = \
                split_data_dictionary["test set"]["preprocessed values"],
            binarised_values = \
                split_data_dictionary["test set"]["binarised values"],
            labels = split_data_dictionary["test set"]["labels"],
            example_names = split_data_dictionary["test set"]["example names"],
            feature_names = split_data_dictionary["feature names"],
            features_mapped = self.features_mapped,
            class_names = split_data_dictionary["class names"],
            feature_selection = self.feature_selection,
            example_filter = self.example_filter,
            preprocessing_methods = self.preprocessing_methods,
            noisy_preprocessing_methods = self.noisy_preprocessing_methods,
            kind = "test"
        )
        
        print(
            "Data sets with {} features{}{}:\n".format(
                training_set.number_of_features,
                " and {} classes".format(self.number_of_classes)
                    if self.number_of_classes else "",
                " ({} superset classes)".format(self.number_of_superset_classes)
                    if self.number_of_superset_classes else ""
                ) +
            "    Training set: {} examples.\n".format(
                training_set.number_of_examples) +
            "    Validation set: {} examples.\n".format(
                validation_set.number_of_examples) +
            "    Test set: {} examples.".format(
                test_set.number_of_examples)
        )
        
        print()
        
        return training_set, validation_set, test_set
    
    def indicesForExampleNames(self, example_names):
        indices = []
        for example_name in example_names:
            index = (self.example_names == example_name).nonzero()[0][0]
            indices.append(index)
        indices = numpy.array(indices)
        return indices
    
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
    
    def clear(self):
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

class SparseRowMatrix(scipy.sparse.csr_matrix):
    def __init__(self, arg1, shape = None, dtype = None, copy = False):
        super(SparseRowMatrix, self).__init__(arg1, shape = shape,
            dtype = dtype, copy = copy)
    
    @property
    def size(self):
        return self.shape[0] * self.shape[1]
    
    def mean(self, axis = None):
        
        if axis is not None:
            return super().mean(axis)
        
        dtype = self.dtype.type
        
        if numpy.issubdtype(dtype, numpy.integer):
            dtype = numpy.float64
        
        self_sum = self.data.sum()
        self_mean = self_sum / self.size
        
        self_mean = self_mean.astype(dtype)
        
        return self_mean
    
    def std(self, axis = None, ddof = 0):
        return numpy.sqrt(self.var(axis = axis, ddof = ddof))
    
    def var(self, axis = None, ddof = 0):
        
        self_squared_mean = self.power(2).mean(axis)
        self_mean_squared = numpy.power(self.mean(axis), 2)
        
        var = self_squared_mean - self_mean_squared
        
        if ddof > 0:
            size = numpy.prod(self.shape)
            var = var * size / (size - ddof)
        
        return var

def standard_deviation(a, axis=None, ddof=0, batch_size=None):
    if not isinstance(a, numpy.ndarray) or axis is not None \
        or batch_size is None:
            return a.std(axis=axis, ddof=ddof)
    return numpy.sqrt(variance(
        a=a,
        axis=axis,
        ddof=ddof,
        batch_size=batch_size
    ))

def variance(a, axis=None, ddof=0, batch_size=None):
    
    if not isinstance(a, numpy.ndarray) or axis is not None \
        or batch_size is None:
            return a.var(axis=axis, ddof=ddof)
    
    number_of_rows = a.shape[0]
    
    mean_squared = numpy.power(a.mean(axis=None), 2)
    
    squared_sum = 0
    
    for i in range(0, number_of_rows, batch_size):
        squared_sum += numpy.power(a[i:i+batch_size], 2).sum()
    
    squared_mean = squared_sum / a.size
    
    var = squared_mean - mean_squared
    
    if ddof > 0:
        size = a.size
        var = var * size / (size - ddof)
    
    return var

def sparsity(a, tolerance = 1e-3, batch_size=None):

    def count_nonzero_values(b):
        return (b >= tolerance).sum()

    if scipy.sparse.issparse(a):
        size = numpy.prod(a.shape)
    else:
        size = a.size

    if batch_size:

        number_of_rows = a.shape[0]

        nonzero_count = 0

        for i in range(0, number_of_rows, batch_size):
            nonzero_count += count_nonzero_values(a[i:i+batch_size])

    else:
        nonzero_count = count_nonzero_values(a)

    a_sparsity = 1 - nonzero_count / size

    return a_sparsity

def parseInput(input_file_or_name):
    
    if input_file_or_name.endswith(".json"):
        
        json_path = input_file_or_name
        
        with open(json_path, "r") as json_file:
            data_set_dictionary = json.load(json_file)
        
        if "title" in data_set_dictionary:
            name = data_set_dictionary["title"]
        else:
            name = baseName(json_path)
        
        if "URLs" not in data_set_dictionary:
            
            if "values" in data_set_dictionary:
                json_directory = os.path.dirname(json_path)
                data_set_dictionary["values"] = os.path.join(
                    json_directory, data_set_dictionary["values"])
            else:
                raise KeyError("Missing path or URL to values.")
            
            if "labels" in data_set_dictionary:
                json_directory = os.path.dirname(json_path)
                data_set_dictionary["labels"] = os.path.join(
                    json_directory, data_set_dictionary["labels"])
    
    elif os.path.isfile(input_file_or_name):
        file_path = input_file_or_name
        name = baseName(file_path)
        data_set_dictionary = {
            "values": file_path
        }
    else:
        name = input_file_or_name
        data_set_dictionary = None
    
    name = normaliseString(name)
    
    return name, data_set_dictionary

def baseName(path):
    base_name = os.path.basename(path)
    base_name = base_name.split(os.extsep, 1)[0]
    return base_name

def saveDataSetDictionaryAsJSONFile(data_set_dictionary, directory):
    
    if "title" in data_set_dictionary:
        name = data_set_dictionary["title"]
    elif "values" in data_set_dictionary:
        name = baseName(data_set_dictionary["values"])
    elif "URLs" in data_set_dictionary:
        name = None
        for key in ["all", "values"]:
            if key in data_set_dictionary["URLs"]:
                if "full" in data_set_dictionary["URLs"]:
                    name = data_set_dictionary["URLs"]["key"]["full"]
                elif "training" in data_set_dictionary["URLs"]:
                    name = data_set_dictionary["URLs"]["key"]["training"]
        if name:
            name = baseName(name)
        else:
            raise Exception("No values found in data set.")
        
    
    name = normaliseString(name)
    json_path = os.path.join(directory, name + ".json")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(json_path, "w") as json_file:
        json.dump(data_set_dictionary, json_file, indent = "\t")

def findDataSet(name, directory):
    
    title = None
    
    for data_set_title in data_sets:
        if normaliseString(data_set_title) == name:
            title = data_set_title
    
    if not title:
        json_path = os.path.join(directory, name, name + ".json")
        if os.path.exists(json_path):
            title, data_set = dataSetFromJSONFile(json_path)
            data_sets[title] = data_set
    
    if not title:
        raise KeyError("Data set not found.")
    
    return title

def dataSetFromJSONFile(json_path):
    
    with open(json_path, "r") as json_file:
        data_set = json.load(json_file)
    
    if "title" in data_set:
        title = data_set["title"]
    else:
        title = baseName(json_path)
    
    if "URLs" not in data_set:
        
        if "values" not in data_set:
            raise Exception(
                "JSON dictionary have to contain either a values entry with "
                "a URL or path to the file containing the value matrix or a "
                "URLs entry containing a dictionary of URLs to files "
                "containing values and optionally labels."
            )
        
        data_set["URLs"] = {
            "values": {
                "full": data_set["values"]
            }
        }
        
        if "labels" in data_set:
            data_set["URLs"]["labels"] = {
                "full": data_set["labels"]
            }
    
    if "loading function" in data_set:
        loading_function_string = data_set["loading function"]
    else:
        loading_function_string = title
    
    data_set["loading function"] = loadingFunction(loading_function_string)
    
    return title, data_set

loading_functions = {
    "default": lambda x: loadMatrixAsDataSet(x, transpose = False),
    "transpose": lambda x: loadMatrixAsDataSet(x, transpose = True),
    "10x": lambda x: load10xDataSet(x),
    "10x-combine": lambda x: loadAndCombine10xDataSets(x),
    "gtex": lambda x: loadGTExDataSet(x)
}

def loadingFunction(search_string):
    
    search_string = normaliseString(search_string)
    
    data_set_loading_function = None
    
    for loading_function_name, loading_function in loading_functions.items():
        if loading_function_name in search_string:
            data_set_loading_function = loading_function
    
    if not data_set_loading_function:
        data_set_loading_function = loading_functions["default"]
    
    return data_set_loading_function

def dataSetTags(title):
    if "tags" in data_sets[title]:
        tags = data_sets[title]["tags"]
    else:
        tags = {
            "example": "example",
            "feature": "feature",
            "mapped feature": "mapped feature",
            "class": "class",
            "type": "value",
            "item": "item"
        }
    
    if "item" in tags and tags["item"]:
        tags["value"] = tags["item"] + " " + tags["type"]
    else:
        tags["value"] = tags["type"]
    
    return tags

def dataSetExampleType(title):
    if "example type" in data_sets[title]:
        return data_sets[title]["example type"]
    else:
        return "unknown"

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

def dataSetClassMapper(title):
    if "class mapping" in data_sets[title]:
        class_mapper = data_sets[title]["class mapping"]
    else:
        class_mapper = None
    return class_mapper

def dataSetClassPalette(title):
    if "class palette" in data_sets[title]:
        class_palette = data_sets[title]["class palette"]
    elif "MNIST" in title:
        index_palette = seaborn.hls_palette(10)
        class_palette = {i: index_palette[i] for i in range(10)}
    elif "10x-PBMC-P" in title:
        classes = data_sets["10x-PBMC-PP"]["URLs"]["all"]
        N = len(classes)
        brewer_palette = seaborn.color_palette("Set3", N)
        class_palette = {c: brewer_palette[i] for i, c in enumerate(classes)}
    else:
        class_palette = None
    return class_palette

def dataSetLabelSuperset(title):
    if "label superset" in data_sets[title]:
        return data_sets[title]["label superset"]
    else:
        return None

def dataSetSortedClassNames(title, superset = False):
    if not superset:
        sorted_class_names = "sorted class names"
    else:
        sorted_class_names = "sorted superset class names"
    
    if sorted_class_names in data_sets[title]:
        return data_sets[title][sorted_class_names]
    else:
        return []

# Default excluded class
# (added later if present in data set labels)
default_excluded_classes = ["No class"]

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
    if "preprocessing methods" in data_sets[title]:
        return data_sets[title]["preprocessing methods"]
    else:
        return None

def acquireDataSet(title, directory):
    
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
                paths[values_or_labels][kind] = None
                continue
            
            URL_filename = os.path.split(URL)[-1]
            possible_extensions = URL_filename.split(os.extsep)
            extensions = []
            
            for possible_extension in reversed(possible_extensions):
                if len(possible_extension) < 8 and possible_extension.isalnum():
                    extensions.insert(0, possible_extension)
                else:
                    break
            
            extension = os.extsep + ".".join(extensions)
            
            filename = "-".join(map(normaliseString,
                [title, values_or_labels, kind]))
            path = os.path.join(directory, filename) + extension
            
            paths[values_or_labels][kind] = path
            
            if not os.path.isfile(path):
                
                if URL.startswith("."):
                    raise Exception("Data set file have to be manually placed "
                        + "in correct folder.")
                if os.path.isfile(URL):
                    
                    print("Copying {} for {} set.".format(
                        values_or_labels, kind, title))
                    start_time = time()
                
                    copyFile(URL, path)
                
                    duration = time() - start_time
                    print("Data set copied ({}).".format(
                        formatDuration(duration)))
                    print()
                    
                else:
                
                    print("Downloading {} for {} set.".format(
                        values_or_labels, kind, title))
                    start_time = time()
                
                    downloadFile(URL, path)
                
                    duration = time() - start_time
                    print("Data set downloaded ({}).".format(
                        formatDuration(duration)))
                    print()
    
    return paths

def loadOriginalDataSet(title, paths):
    
    print("Loading original data set.")
    loading_time_start = time()
    
    data_dictionary = data_sets[title]["loading function"](paths)
    
    loading_duration = time() - loading_time_start
    print("Original data set loaded ({}).".format(formatDuration(
        loading_duration)))
    
    if not isinstance(data_dictionary["values"], scipy.sparse.csr_matrix):
        
        print()
    
        print("Converting data set value array to sparse matrix.")
        sparse_time_start = time()
        
        data_dictionary["values"] = scipy.sparse.csr_matrix(
            data_dictionary["values"])
        
        sparse_duration = time() - sparse_time_start
        print("Data set value array converted ({}).".format(formatDuration(
            sparse_duration)))
    
    return data_dictionary

def preprocessedPathFunction(preprocess_directory = "", name = ""):
    
    def preprocessedPath(base_name = None, map_features = None,
        preprocessing_methods = None,
        feature_selection = None, feature_selection_parameters = None,
        example_filter = None, example_filter_parameters = None,
        splitting_method = None, splitting_fraction = None,
        split_indices = None):
        
        base_path = os.path.join(preprocess_directory, name)
        
        filename_parts = [base_path]
        
        if base_name:
            filename_parts.append(normaliseString(base_name))
        
        if map_features:
            filename_parts.append("features_mapped")
        
        if feature_selection:
            feature_selection_part = normaliseString(feature_selection)
            if feature_selection_parameters:
                for parameter in feature_selection_parameters:
                    feature_selection_part += "_" + normaliseString(str(
                        parameter))
            filename_parts.append(feature_selection_part)
        
        if example_filter:
            example_filter_part = normaliseString(example_filter)
            if example_filter_parameters:
                for parameter in example_filter_parameters:
                    example_filter_part += "_" + normaliseString(str(
                        parameter))
            filename_parts.append(example_filter_part)
        
        if preprocessing_methods:
            filename_parts.extend(map(normaliseString, preprocessing_methods))
        
        
        if splitting_method:
            filename_parts.append("split")
            
            if splitting_method == "indices" and \
                len(split_indices) == 3 or not splitting_fraction:
                
                filename_parts.append(splitting_method)
            else:
                filename_parts.append("{}_{}".format(
                    splitting_method,
                    splitting_fraction
                ))
        
        path = "-".join(filename_parts) + preprocessed_extension
        
        return path
    
    return preprocessedPath

def updateTagForMappedFeatures(tags):
    
    mapped_feature_tag = tags.pop("mapped feature", None)
    
    if mapped_feature_tag:
        tags["feature"] = mapped_feature_tag
    
    return tags

def mapFeatures(values, feature_IDs, feature_mapping):
    
    values = scipy.sparse.csc_matrix(values)
    
    M, N_IDs = values.shape
    N_features = len(feature_mapping)
    
    feature_name_from_ID = {
        v: k for k, vs in feature_mapping.items() for v in vs
    }
    
    N_unknown_IDs = 0
    
    for feature_ID in feature_IDs:
        if feature_ID not in feature_name_from_ID:
            feature_name_from_ID[feature_ID] = feature_ID
            N_unknown_IDs += 1
    
    if N_unknown_IDs > 0:
        print("{0} feature{1} cannot be mapped -- using original feature{1}."\
            .format(N_unknown_IDs, "s" if N_unknown_IDs > 1 else ""))
    
    N_features += N_unknown_IDs
    
    aggregated_values = numpy.zeros((M, N_features), values.dtype)
    feature_names_with_index = dict()
    
    for i, feature_ID in enumerate(feature_IDs):
        
        feature_name = feature_name_from_ID[feature_ID]
        
        if feature_name in feature_names_with_index:
            index = feature_names_with_index[feature_name]
        else:
            index = len(feature_names_with_index)
            feature_names_with_index[feature_name] = index
        
        aggregated_values[:, index] += values[:, i].A.flatten()
    
    feature_names = list(feature_names_with_index.keys())
    
    feature_names_not_found = set(feature_mapping.keys()) - set(feature_names)
    N_feature_names_not_found = len(feature_names_not_found)
    N_features -= N_feature_names_not_found
    aggregated_values = aggregated_values[:, :N_features]
    
    if N_feature_names_not_found > 0:
        print(
            "Did not find any original features for {} new feature{}.".format(
                N_feature_names_not_found,
                "s" if N_feature_names_not_found > 1 else ""
            )
        )
    
    aggregated_values = SparseRowMatrix(aggregated_values)
    feature_names = numpy.array(feature_names)
    
    return aggregated_values, feature_names

def selectFeatures(values_dictionary, feature_names, feature_selection = None,
    feature_selection_parameters = None, preprocessPath = None):
    
    feature_selection = normaliseString(feature_selection)
    
    print("Selecting features.")
    start_time = time()
    
    if type(values_dictionary) == dict:
        values = values_dictionary["original"]
    
    M, N = values.shape
    
    if feature_selection == "remove_zeros":
        total_feature_sum = values.sum(axis = 0)
        if isinstance(total_feature_sum, numpy.matrix):
            total_feature_sum = total_feature_sum.A.squeeze()
        indices = total_feature_sum != 0
    
    elif feature_selection == "keep_gini_indices_above":
        gini_indices = loadWeights(values, "gini", preprocessPath)
        if feature_selection_parameters:
            threshold = float(feature_selection_parameters[0])
        else:
            threshold = 0.1
        indices = gini_indices > threshold

    elif feature_selection == "keep_highest_gini_indices":
        gini_indices = loadWeights(values, "gini", preprocessPath)
        gini_sorted_indices = numpy.argsort(gini_indices)
        if feature_selection_parameters:
            number_to_keep = int(feature_selection_parameters[0])
        else:
            number_to_keep = int(M/2)
        indices = numpy.sort(gini_sorted_indices[-number_to_keep:])
        
    elif feature_selection == "keep_variances_above":
        variances = values.var(axis = 0)
        if isinstance(variances, numpy.matrix):
            variances = variances.A.squeeze()
        if feature_selection_parameters:
            threshold = float(feature_selection_parameters[0])
        else:
            threshold = 0.5
        indices = variances > threshold

    elif feature_selection == "keep_highest_variances":
        variances = values.var(axis = 0)
        if isinstance(variances, numpy.matrix):
            variances = variances.A.squeeze()
        variance_sorted_indices = numpy.argsort(variances)
        if feature_selection_parameters:
            number_to_keep = int(feature_selection_parameters[0])
        else:
            number_to_keep = int(M/2)
        indices = numpy.sort(variance_sorted_indices[-number_to_keep:])
        
    else:
        indices = numpy.arange(N)
    
    if feature_selection:
        error = ValueError(
            "No features excluded using feature selection. Exiting.")
        if indices.dtype == "bool" and all(indices):
            raise error
        elif indices.dtype != "bool" and len(indices) == N:
            raise error
    
    feature_selected_values = {}
    
    for version, values in values_dictionary.items():
        if values is not None:
            feature_selected_values[version] = values[:, indices]
        else:
            feature_selected_values[version] = None
    
    feature_selected_feature_names = feature_names[indices]
    
    N_changed = len(feature_selected_feature_names)
    
    duration = time() - start_time
    print("{} features selected, {} excluded ({}).".format(
        N_changed,
        N - N_changed,
        formatDuration(duration)
    ))
    
    return feature_selected_values, feature_selected_feature_names

def defaultFeatureParameters(feature_selection = None,
    number_of_features = None):
    
    if feature_selection:
        feature_selection = normaliseString(feature_selection)
        M = number_of_features
        
        if feature_selection == "remove_zeros":
            feature_selection_parameters = None
        
        elif feature_selection == "keep_gini_indices_above":
            feature_selection_parameters = [0.1]
        
        elif feature_selection == "keep_highest_gini_indices":
            feature_selection_parameters = [int(M/2)]
        
        elif feature_selection == "keep_variances_above":
            feature_selection_parameters = [0.5]
        
        elif feature_selection == "keep_highest_variances":
            feature_selection_parameters = [int(M/2)]
        
        else:
            feature_selection_parameters = None
    
    else:
        feature_selection_parameters = None
    
    return feature_selection_parameters

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
    elif labels is not None:
        filter_labels = labels.copy()
        filter_excluded_classes = excluded_classes
    else:
        filter_labels = None
    
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
    
    elif example_filter == "inverse_macosko":
        maximum_number_of_non_zero_elements = 900
        number_of_non_zero_elements = (values != 0).sum(axis = 1)
        filter_indices = numpy.nonzero(
            number_of_non_zero_elements <= maximum_number_of_non_zero_elements
        )[0]
    
    elif example_filter in ["keep", "remove", "excluded_classes"]:
        
        if filter_labels is None:
            raise ValueError("Cannot filter examples based on labels, "
                "since data set is unlabelled.")
        
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
    
    if example_filter and len(filter_indices) == M:
        raise ValueError("No examples filtered out using example filter. Exiting.")
    
    example_filtered_values = {}
    
    for version, values in values_dictionary.items():
        if values is not None:
            example_filtered_values[version] = values[filter_indices, :]
        else:
            example_filtered_values[version] = None
    
    example_filtered_example_names = example_names[filter_indices]
    
    if labels is not None:
        example_filtered_labels = labels[filter_indices]
    else:
        example_filtered_labels = None
    
    M_changed = len(example_filtered_example_names)
    
    duration = time() - start_time
    print("{} examples filtered out, {} remaining ({}).".format(
        M - M_changed,
        M_changed,
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
        
        elif preprocessing_method == "log":
            preprocess = lambda values: values.logp1()
        
        elif preprocessing_method == "exp":
            preprocess = lambda values: values.expm1()
        
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
    
    random_state = numpy.random.RandomState(42)
    
    if method == "random":
        
        M_training_validation = int(fraction * M)
        M_training = int(fraction * M_training_validation)
        
        shuffled_indices = random_state.permutation(M)
        
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
    
    elif method == "macosko":
        
        values = data_dictionary["values"]
        
        minimum_number_of_non_zero_elements = 900
        number_of_non_zero_elements = (values != 0).sum(axis = 1)
        
        training_indices = numpy.nonzero(
            number_of_non_zero_elements > minimum_number_of_non_zero_elements
        )[0]
        
        test_validation_indices = numpy.nonzero(
            number_of_non_zero_elements <= minimum_number_of_non_zero_elements
        )[0]
        
        random_state.shuffle(test_validation_indices)
        
        N_test_validation = len(test_validation_indices)
        V = int((1 - fraction) * N_test_validation)
        
        validation_indices = test_validation_indices[:V]
        test_indices = test_validation_indices[V:]
    
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
        "feature names": data_dictionary["feature names"],
        "class names": data_dictionary["class names"]
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
    
    return split_data_dictionary

def decomposeDataSubsets(*subsets, method=None, number_of_components=None,
                      random=False):
    
    # Check input
    
    titles = set()
    versions = set()
    kinds = {}
    
    for subset in subsets:
        titles.add(subset.title)
        versions.add(subset.version)
        kinds.setdefault(subset.kind, [])
        kinds[subset.kind].append(subset)
    
    if len(titles) > 1:
        raise ValueError(
            "Data subsets are not from the same data set. " +
            "Subsets provided are from: " +
            ", ".join(titles) + "."
        )
    
    if len(versions) > 1:
        raise ValueError(
            "Data subsets does not have the same version. " +
            "Subsets provided are: " +
            ", ".join(versions) + "."
        )
    
    for kind, kind_subsets in kinds.items():
        
        if kind not in subset_kinds:
            raise ValueError(
                "Subset of unknown kind provided: {}".format(kind)
            )
        
        if len(kind_subsets) > 1:
            
            kind_subset = kind_subsets.pop()
            kind_values = kind_subset.values
            
            for other_kind_subset in kind_subsets:
                other_kind_values = other_kind_subset.values
                if (other_kind_values != kind_values).any():
                    raise ValueError(
                        "Multiple {} subsets represented with ".format(kind) +
                        "different values."
                    )
    
    version = versions.pop()
    
    # Setup
    
    if method is None:
        method = "pca"
    else:
        method = normaliseString(method)
    
    method = properString(method, DECOMPOSITION_METHOD_NAMES)
    unique_subset_names = set(s.kind for s in subsets)
    
    print(
        "Decomposing {} values for {} set{} using {}-d {}.".format(
            version,
            enumerateListOfStrings(list(unique_subset_names))
                if "full" not in unique_subset_names
                else "full",
            "s" if len(unique_subset_names) > 1
                and "full" not in unique_subset_names
                else "",
            number_of_components,
            method
        )
    )
    time_start = time()
    
    # Extract values
    
    full_set = None
    unique_subsets = []
    
    for subset in subsets:
        if not full_set and subset.kind == "full":
            full_set = subset
        else:
            unique_subsets.append(subset)
    
    if full_set:
        values = full_set.values.copy()
        
        kind_indices = {}
        
        for subset in unique_subsets:
            subset_indices = full_set.indicesForExampleNames(
                subset.example_names)
            kind_indices[subset.kind] = subset_indices
            values[subset_indices] = subset.values
        
    else:
        values = numpy.vstack([s.values for s in unique_subsets])
    
    # Decompose values
    
    decomposed_values = decompose(
        values=values,
        method=method,
        number_of_components=number_of_components,
        random=random
    )
    
    # Package results
    
    decomposed_value_sets = {}
    
    if full_set:
        decomposed_value_sets["full"] = decomposed_values
    
        for subset in unique_subsets:
            subset_indices = kind_indices[subset.kind]
            decomposed_value_sets[subset.kind] \
                = decomposed_values[subset_indices]
    
    else:
        cursor = 0
        
        for subset in unique_subsets:
            M = len(subset.values)
            subset_indices = numpy.arange(M) + cursor
            decomposed_value_sets[subset.kind] \
                = decomposed_values[subset_indices]
            cursor += M
    
    decomposition_label = DECOMPOSITION_METHOD_LABEL[method]
    feature_names = numpy.array([
        "{}{}".format(decomposition_label, j + 1)
        for j in range(number_of_components)
    ])
    
    decomposed_subsets = []
    
    for subset in subsets:
        decomposed_subset = DataSet(
            subset.name,
            values = decomposed_value_sets[subset.kind],
            preprocessed_values = None,
            binarised_values = None,
            labels = subset.labels,
            example_names = subset.example_names,
            feature_names = feature_names,
            features_mapped = subset.features_mapped,
            class_names = subset.class_names,
            feature_selection = subset.feature_selection,
            example_filter = subset.example_filter,
            preprocessing_methods = method,
            noisy_preprocessing_methods = None,
            kind = subset.kind
        )
        decomposed_subsets.append(decomposed_subset)
    
    if len(subsets) == 1:
        decomposed_subsets = decomposed_subsets.pop()
    
    duration = time() - time_start
    print("Values decomposed ({}).".format(formatDuration(duration)))
    
    return decomposed_subsets

def loadDataDictionary(path):
    
    def load(tables_file, group = None):
        
        if not group:
            group = tables_file.root
        
        data_dictionary = {}
        
        for node in tables_file.iter_nodes(group):
            node_title = node._v_title
            if node == group:
                pass
            elif isinstance(node, tables.Group):
                if node_title.endswith("set"):
                    data_dictionary[node_title] = load(
                        tables_file, group = node)
                elif node_title.endswith("values"):
                    data_dictionary[node_title] = loadSparseMatrix(
                        tables_file, group = node)
                elif node_title == "split indices":
                    data_dictionary[node_title] = loadSplitIndices(
                        tables_file, group = node)
                elif node_title == "feature mapping":
                    data_dictionary[node_title] = loadFeatureMapping(
                        tables_file, group = node)
                else:
                    raise NotImplementedError(
                        "Loading group `{}` not implemented.".format(
                            node_title)
                    )
            elif isinstance(node, tables.Array):
                data_dictionary[node_title] = loadArrayOrOtherType(node)
            else:
                raise NotImplementedError(
                    "Loading node `{}` not implemented.".format(node_title)
                )
        
        return data_dictionary
    
    start_time = time()
    
    with tables.open_file(path, "r") as tables_file:
        data_dictionary = load(tables_file)
    
    duration = time() - start_time
    print("Data loaded ({}).".format(formatDuration(duration)))
    
    return data_dictionary

def loadArrayOrOtherType(node):
    
    value = node.read()
    
    if value.dtype.char == "S":
        decode = numpy.vectorize(lambda s: s.decode("UTF-8"))
        value = decode(value).astype("U")
    
    elif value.dtype == numpy.uint8:
        value = value.tostring().decode("UTF-8")
        
        if value == "None":
            value = None
    
    if node._v_name.endswith("_was_list"):
        value = value.tolist()
    
    return value

def loadSparseMatrix(tables_file, group):
    
    arrays = {}
    
    for array in tables_file.iter_nodes(group, "Array"):
        arrays[array.title] = array.read()
    
    sparse_matrix = scipy.sparse.csr_matrix(
        (arrays["data"], arrays["indices"], arrays["indptr"]),
        shape = arrays["shape"]
    )
    
    return sparse_matrix

def loadSplitIndices(tables_file, group):
    
    split_indices = {}
    
    for array in tables_file.iter_nodes(group, "Array"):
        start, stop = array.read()
        split_indices[array.title] = slice(start, stop)
    
    return split_indices

def loadFeatureMapping(tables_file, group):
    
    feature_lists = {}
    
    for array in tables_file.iter_nodes(group, "Array"):
        feature_lists[array.title] = array.read().tolist()
    
    feature_names = feature_lists["feature_names"]
    feature_counts = feature_lists["feature_counts"]
    feature_IDs = feature_lists["feature_IDs"]
    
    feature_mapping = {}
    
    for feature_name, feature_count in zip(feature_names, feature_counts):
        feature_name = feature_name.decode("UTF-8")
        feature_ID_set = [
            feature_IDs.pop(0).decode("UTF-8") for i in range(feature_count)
        ]
        feature_mapping[feature_name] = feature_ID_set
    
    return feature_mapping

def saveDataDictionary(data_dictionary, path):
    
    directory, filename = os.path.split(path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    def save(data_dictionary, tables_file, group_title = None):
        
        if group_title:
            group = tables_file.create_group("/",
                normaliseString(group_title), group_title)
        else:
            group = tables_file.root
        
        for title, value in data_dictionary.items():
            
            if isinstance(value, scipy.sparse.csr_matrix):
                saveSparseMatrix(value, title, group, tables_file)
            elif isinstance(value, (numpy.ndarray, list)):
                saveArray(value, title, group, tables_file)
            elif title == "split indices":
                saveSplitIndices(value, title, group, tables_file)
            elif title == "feature mapping":
                saveFeatureMapping(value, title, group, tables_file)
            elif value is None:
                saveString(str(value), title, group, tables_file)
            elif title.endswith("set"):
                save(value, tables_file, group_title = title)
            else:
                raise NotImplementedError(
                    "Saving type {} for title \"{}\" has not been implemented."
                        .format(type(value), title)
                )
    
    start_time = time()
    
    filters = tables.Filters(complib = "zlib", complevel = 5)
    
    with tables.open_file(path, "w", filters = filters) as tables_file:
        save(data_dictionary, tables_file)
    
    duration = time() - start_time
    print("Data saved ({}).".format(formatDuration(duration)))

def saveArray(array, title, group, tables_file):
    name = normaliseString(title)
    if isinstance(array, list):
        array = numpy.array(array)
        name += "_was_list"
    if array.dtype.char == "U":
        encode = numpy.vectorize(lambda s: s.encode("UTF-8"))
        array = encode(array).astype("S")
    atom = tables.Atom.from_dtype(array.dtype)
    data_store = tables_file.create_carray(
        group,
        name,
        atom,
        array.shape,
        title
    )
    data_store[:] = array

def saveString(string, title, group, tables_file):
    encoded_string = numpy.frombuffer(string.encode('UTF-8'), numpy.uint8)
    saveArray(encoded_string, title, group, tables_file)

def saveSparseMatrix(sparse_matrix, title, group, tables_file):
    
    name = normaliseString(title)
    group = tables_file.create_group(group, name, title)
    
    for attribute in ("data", "indices", "indptr", "shape"):
        array = numpy.array(getattr(sparse_matrix, attribute))
        saveArray(array, attribute, group, tables_file)

def saveSplitIndices(split_indices, title, group, tables_file):
    
    name = normaliseString(title)
    group = tables_file.create_group(group, name, title)
    
    for subset_name, subset_slice in split_indices.items():
        subset_slice_array = numpy.array(
            [subset_slice.start, subset_slice.stop])
        saveArray(subset_slice_array, subset_name, group, tables_file)

def saveFeatureMapping(feature_mapping, title, group, tables_file):
    
    name = normaliseString(title)
    group = tables_file.create_group(group, name, title)
    
    feature_names = []
    feature_counts = []
    feature_IDs = []
    
    for feature_name, feature_ID_set in feature_mapping.items():
        feature_names.append(feature_name)
        feature_counts.append(len(feature_ID_set))
        feature_IDs.extend(feature_ID_set)
    
    feature_lists = {
        "feature_names": feature_names,
        "feature_counts": feature_counts,
        "feature_IDs": feature_IDs
    }
    
    for feature_list_name, feature_list in feature_lists.items():
        feature_list_array = numpy.array(feature_list)
        saveArray(feature_list_array, feature_list_name, group, tables_file)

def loadMouseRetinaDataSet(paths):
    
    values, column_headers, row_indices = \
        loadTabSeparatedMatrix(paths["values"]["full"], numpy.float32)
    
    values = values.T
    example_names = numpy.array(column_headers)
    
    feature_column = 0
    feature_names = numpy.array(row_indices)[:, feature_column]
    
    if paths["labels"]["full"]:
        labels = loadLabelsFromDelimiterSeparetedValues(
            path = paths["labels"]["full"],
            label_column = 1,
            example_column = 0,
            example_names = example_names,
            header = None,
            dtype = numpy.int32,
            default_label = 0
        )
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    return data_dictionary

def load10xDataSet(paths):
    
    data_dictionary = loadValuesFrom10xDataSet(paths["values"]["full"])
    values = data_dictionary["values"]
    example_names = data_dictionary["example names"]
    feature_names = data_dictionary["feature names"]
    
    if paths["labels"]["full"]:
        labels = loadLabelsFromDelimiterSeparetedValues(
            path = paths["labels"]["full"],
            label_column = "celltype",
            example_column = "barcodes",
            example_names = example_names,
            dtype = "U"
        )
    else:
        labels = None
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    return data_dictionary

def loadAndCombine10xDataSets(paths):
    
    # Initialisation
    
    value_sets = {}
    example_name_sets = {}
    feature_name_sets = {}
    genome_names = {}
    
    # Loading values from separate data sets
    
    for class_name, path in paths["all"].items():
        data_dictionary = loadValuesFrom10xDataSet(path)
        value_sets[class_name] = data_dictionary["values"]
        example_name_sets[class_name] = data_dictionary["example names"]
        feature_name_sets[class_name] = data_dictionary["feature names"]
        genome_names[class_name] = data_dictionary["genome name"]
    
    # Check for multiple genomes
    
    class_name, genome_name = genome_names.popitem()
    
    for other_class_names, other_genome_name in genome_names.items():
        if not genome_name == other_genome_name:
            raise ValueError(
                "The genome names for \"{}\" and \"{}\" do not match."
                    .format(class_name, other_class_name)
            )
    
    # Infer labels
    
    label_sets = {}
    
    for class_name in example_name_sets:
        label_sets[class_name] = numpy.array([class_name] \
            * example_name_sets[class_name].shape[0])
    
    # Combine data sets
    
    sorted_values = lambda d: [v for k, v in sorted(d.items())]
    
    values = scipy.sparse.vstack(sorted_values(value_sets))
    example_names = numpy.concatenate(sorted_values(example_name_sets))
    labels = numpy.concatenate(sorted_values(label_sets))
    
    # Extract feature names and check for differences
    
    class_name, feature_names = feature_name_sets.popitem()
    
    for other_class_name, other_feature_names in feature_name_sets.items():
        if not all(feature_names == other_feature_names):
            raise ValueError(
                "The feature names for \"{}\" and \"{}\" do not match."
                    .format(class_name, other_class_name)
            )
    
    # Return data
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
    }
    
    return data_dictionary

def loadValuesFrom10xDataSet(path):
    
    parent_paths = set()
    
    multiple_directories_error = NotImplementedError(
        "Cannot handle data sets with multiple directories."
    )
    
    if path.endswith(".h5"):
        with tables.open_file(path, "r") as f:
            
            table = {}
            
            for node in f.walk_nodes(where="/", classname="Array"):
                node_path = node._v_pathname
                parent_path, node_name = os.path.split(node_path)
                parent_paths.add(parent_path)
                if len(parent_paths) > 1:
                    raise multiple_directories_error
                table[node.name] = node.read()
            
            values = scipy.sparse.csc_matrix(
                (table["data"], table["indices"], table["indptr"]),
                shape=table["shape"]
            )
            
            example_names = table["barcodes"]
            feature_names = table["gene_names"]
    
    elif path.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tarball:
            for member in sorted(tarball, key = lambda member: member.name):
                if member.isfile():
                    
                    parent_path, filename = os.path.split(member.name)
                    parent_paths.add(parent_path)
                    
                    if len(parent_paths) > 1:
                        raise multiple_directories_error
                    
                    name, extension = os.path.splitext(filename)
                    
                    with tarball.extractfile(member) as data_file:
                        if filename == "matrix.mtx":
                            values = scipy.io.mmread(data_file)
                        elif extension == ".tsv":
                            names = numpy.array(data_file.read().splitlines())
                            if name == "barcodes":
                                example_names = names
                            elif name == "genes":
                                feature_names = names
    
    values = values.T
    example_names = example_names.astype("U")
    feature_names = feature_names.astype("U")
    
    if len(parent_paths) == 1:
        parent_path = parent_paths.pop()
    else:
        raise multiple_directories_error
    
    _, genome_name = os.path.split(parent_path)
    
    data_dictionary = {
        "values": values,
        "example names": example_names,
        "feature names": feature_names,
        "genome name": genome_name
    }
    
    return data_dictionary

def loadTCGADataSet(paths):
    
    # Values, example names, and feature names
    
    values, column_headers, row_indices = \
        loadTabSeparatedMatrix(paths["values"]["full"], numpy.float32)

    values = values.T
    values = numpy.power(2, values) - 1
    values = numpy.round(values)

    example_names = numpy.array(column_headers)

    feature_ID_column = 0
    feature_IDs = numpy.array(row_indices)[:, feature_ID_column]
    
    # Labels
    
    if paths["labels"]["full"]:
        labels = loadLabelsFromDelimiterSeparetedValues(
            path = paths["labels"]["full"],
            label_column = "_primary_site",
            example_column = "sampleID",
            example_names = example_names,
            dtype = "U"
        )
    
    # Feature mapping
    
    feature_mapping = dict()
    
    with gzip.open(paths["feature mapping"]["full"], "rt") \
        as feature_mapping_file:

        for row in feature_mapping_file:
            if row.startswith("#"):
                continue
            row_elements = row.split()
            feature_name = row_elements[1]
            feature_ID = row_elements[0]
            if feature_name not in feature_mapping:
                feature_mapping[feature_name] = []
            feature_mapping[feature_name].append(feature_ID)
    
    # Dictionary
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_IDs,
        "feature mapping": feature_mapping
    }
    
    return data_dictionary

def loadGTExDataSet(paths):
    
    # Values, example names and feature names
    
    values, column_headers, row_indices = \
        loadTabSeparatedMatrix(paths["values"]["full"], numpy.float32)

    values = values.T

    example_names = numpy.array(column_headers)

    feature_ID_column = 0
    feature_name_column = 1
    
    feature_IDs = numpy.array(row_indices)[:, feature_ID_column]
    feature_names = numpy.array(row_indices)[:, feature_name_column]
    
    # Labels
    
    if paths["labels"]["full"]:
        labels = loadLabelsFromDelimiterSeparetedValues(
            path = paths["labels"]["full"],
            label_column = "SMTSD",
            example_column = "SAMPID",
            example_names = example_names,
            dtype = "U"
        )
    
    # Feature mapping
    
    feature_mapping = dict()
    
    for feature_name, feature_ID in zip(feature_names, feature_IDs):
        if feature_name not in feature_mapping:
            feature_mapping[feature_name] = []
        feature_mapping[feature_name].append(feature_ID)
    
    # Dictionary
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_IDs,
        "feature mapping": feature_mapping
    }
    
    return data_dictionary

def loadMatrixAsDataSet(paths, transpose = True):
    
    # Values
    
    values, column_headers, row_indices = \
        loadTabSeparatedMatrix(paths["values"]["full"], numpy.float32)
    
    if transpose:
        values = values.T
        example_names = numpy.array(column_headers)
        feature_names = numpy.array(row_indices)
    else:
        example_names = numpy.array(row_indices)
        feature_names = numpy.array(column_headers)
    
    example_names = example_names.flatten()
    feature_names = feature_names.flatten()
    
    # Labels
    
    if "labels" in paths:
        labels = loadLabelsFromDelimiterSeparetedValues(
            path=paths["labels"]["full"],
            example_names=example_names
        )
    else:
        labels = None
    
    # Result
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }
    
    return data_dictionary

def loadTabSeparatedMatrix(tsv_path, data_type = None):
    
    tsv_extension = tsv_path.split(os.extsep, 1)[-1]
    
    if tsv_extension == "tsv":
        openFile = lambda path: open(path, "rt")
    elif tsv_extension.endswith("gz"):
        openFile = lambda path: gzip.open(path, "rt")
    else:
        raise NotImplementedError(
            "Loading from file with extension `{}` not implemented.".format(
                tsv_extension)
        )
    
    values = []
    row_indices = []
    
    with openFile(tsv_path) as tsv_file:
        
        column_headers = None
        
        while not column_headers:
            
            row_elements = next(tsv_file).split()
            
            # Skip, if row could not be split into elements
            if len(row_elements) <= 1:
                continue
            
            # Skip, if row only contains two integers before header
            # (assumed to be the shape of the matrix)
            elif len(row_elements) == 2 \
                and all([element.isdigit() for element in row_elements]):
                continue
            
            column_headers = row_elements
        
        row_elements = next(tsv_file).split()
        
        for i, element in enumerate(row_elements):
            if isfloat(element):
                column_offset = i
                break
        
        column_header_offset = column_offset - (
            len(row_elements) - len(column_headers)
        )
        
        column_headers = column_headers[column_header_offset:]
        
        def parseRowElements(row_elements):
            row_index = row_elements[:column_offset]
            row_indices.append(row_index)
            row_values = list(map(float, row_elements[column_offset:]))
            values.append(row_values)
        
        parseRowElements(row_elements)
        
        for row in tsv_file:
            parseRowElements(row.split())
    
    values = numpy.array(values, data_type)
    
    return values, column_headers, row_indices

def loadLabelsFromDelimiterSeparetedValues(path, label_column = 1,
    example_column = 0, example_names = None, delimiter = None,
    header = "infer", dtype = None, default_label = "No class"):
    
    if not delimiter:
        if path.endswith(".csv"):
            delimiter = ","
        else:
            delimiter = "\t"
    
    metadata = pandas.read_csv(
        path,
        index_col = example_column,
        usecols = [example_column, label_column],
        delimiter = delimiter,
        header = header
    )
    
    if isinstance(label_column, int):
        label_column = metadata.columns[0]
    
    unordered_labels = metadata[label_column]
    
    if example_names is not None:
        
        labels = numpy.zeros(example_names.shape, unordered_labels.dtype)
        labels[labels == 0] = default_label
        
        for example_name, label in unordered_labels.items():
            labels[example_names == example_name] = label
    
    else:
        labels = unordered_labels.values
    
    if dtype is None and labels.dtype == "object":
        dtype = "U"
    
    if dtype:
        labels = labels.astype(dtype)
    
    return labels

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
    
    values = values.astype(numpy.float32)
    
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
        values[kind] = numpy.loadtxt(paths["values"][kind], numpy.float32)
    
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
    
    random_state = numpy.random.RandomState(60)
    
    M = number_of_examples
    N = number_of_features
    
    values = numpy.empty((M, N), numpy.float32)
    labels = numpy.empty(M, numpy.int32)
    
    r = numpy.empty((M, N))
    p = numpy.empty((M, N))
    dropout = numpy.empty((M, N))
    
    r_draw = lambda: scale * random_state.rand(N)
    p_draw = lambda: random_state.rand(N)
    dropout_draw = lambda: random_state.rand(N)
    
    r_type = r_draw()
    p_type = p_draw()
    dropout_type = dropout_draw()
    
    label = 1
    
    for i in range(M):
        u = random_state.rand()
        if u > 1 - update_probability:
            r_type = r_draw()
            p_type = p_draw()
            dropout_type = dropout_draw()
            label += 1
        r[i] = r_type
        p[i] = p_type
        dropout[i] = dropout_type
        labels[i] = label
    
    shuffled_indices = random_state.permutation(M)

    r = r[shuffled_indices]
    p = p[shuffled_indices]
    dropout = dropout[shuffled_indices]
    labels = labels[shuffled_indices]
    
    no_class_indices = random_state.permutation(M)[:int(0.1 * M)]
    labels[no_class_indices] = 0
    
    for i in range(M):
        for j in range(N):
            # value = random_state.poisson(r[i, j])
            value = random_state.negative_binomial(r[i, j], p[i, j])
            value_dropout = random_state.binomial(1, dropout[i, j])
            # value_dropout = random_state.poisson(dropout[i, j])
            values[i, j] = value_dropout * value
    
    example_names = numpy.array(["example {}".format(i + 1) for i in range(M)])
    feature_IDs = numpy.array(["feature {}".format(j + 1) for j in range(N)])
    
    feature_names = ["feature " + n for n in "ABCDE"]
    feature_ID_groups = numpy.split(feature_IDs, len(feature_names))
    
    feature_mapping = {
        feature_name: feature_ID_group.tolist()
        for feature_name, feature_ID_group in
        zip(feature_names, feature_ID_groups)
    }
    
    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_IDs,
        "feature mapping": feature_mapping
    }
    
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
    
    return data.multiply(weights)

def loadWeights(data, method, preprocessPath):
    
    if preprocessPath:
        weights_path = preprocessPath(method + "-weights")
    else:
        weights_path = None
    
    if weights_path and os.path.isfile(weights_path):
        print("Loading weights from.")
        weights_dictionary = loadDataDictionary(weights_path)
    else:
        start_time = time()
        
        if method == "gini":
            weights = computeGiniIndices(data)
        elif method == "idf":
            weights = computeInverseGlobalFrequencyWeights(data)
        
        duration = time() - start_time
        
        weights_dictionary = {"weights": weights}
        
        if weights_path and duration > maximum_duration_before_saving:
            print("Saving weights.")
            saveDataDictionary(weights_dictionary, weights_path)
    
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
    
    gini_indices = numpy.zeros(N)
    
    for i in range(0, N, batch_size):
        batch = data[:, i:(i+batch_size)].A
        
        # Values cannot be 0
        batch = numpy.clip(batch, epsilon, batch)
        
        # Array should be normalised and sorted frequencies over the examples
        batch = numpy.sort(batch / (numpy.sum(batch, axis = 0)), axis = 0)
        
        # Gini coefficients over the examples for each feature. 
        gini_indices[i:(i+batch_size)] = index_vector @ batch / M
    
    duration = time() - start_time
    print("Gini indices computed ({}).".format(formatDuration(duration)))
    
    return gini_indices

def computeInverseGlobalFrequencyWeights(data):
    
    print("Computing IDF weights.")
    start_time = time()
    
    M = data.shape[0]
    
    global_frequencies = data.astype(bool).sum(axis=0).A.squeeze()
    
    idf_weights = numpy.log(M / (global_frequencies + 1))
    
    duration = time() - start_time
    print("IDF weights computed ({}).".format(formatDuration(duration)))
    
    return idf_weights

def supersetLabels(labels, label_superset):
    
    if not label_superset:
        superset_labels = None
    
    elif label_superset == "infer":
        superset_labels = []

        for label in labels:
            superset_label = re.match("^( ?[A-Za-z])+", label).group()
            superset_labels.append(superset_label)
        
        superset_labels = numpy.array(superset_labels)
    
    else:
        label_superset_reverse = {v: k for k, vs in label_superset.items()
            for v in vs}
            
        label_to_superset_label = lambda label: label_superset_reverse[label]
        labels_to_superset_labels = numpy.vectorize(label_to_superset_label)
        
        superset_labels = labels_to_superset_labels(labels)
    
    return superset_labels

def supersetClassPalette(class_palette, label_superset):
    
    if not label_superset or label_superset == "infer":
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

GENERAL_CLASS_NAMES = ["Others", "Unknown", "No class", "Remaining"]

def createLabelSorter(sorted_class_names = []):
    
    def labelSorter(label):
        
        label = str(label)
        
        K = len(sorted_class_names)
        L = len(GENERAL_CLASS_NAMES)
        index_width = len(str(K+L))
        
        if label in sorted_class_names:
            index = sorted_class_names.index(label)
        elif label in GENERAL_CLASS_NAMES:
            index = K + GENERAL_CLASS_NAMES.index(label)
        else:
            index = K + L
        
        label =  "{:{}d} {}".format(index, index_width, label)
        
        return label
    
    return labelSorter

def directory(base_directory, data_set, splitting_method, splitting_fraction,
    preprocessing = True):
    
    data_set_directory = os.path.join(base_directory, data_set.name)
    
    # Splitting directory
    
    if splitting_method:
        
        splitting_directory_parts = ["split"]
    
        if splitting_method == "default":
            splitting_method = data_set.defaultSplittingMethod()
    
        if splitting_method == "indices" and len(data_set.split_indices) == 3 \
            or not splitting_fraction:
        
            splitting_directory_parts.append(splitting_method)
        else:
            splitting_directory_parts.append("{}_{}".format(splitting_method,
            splitting_fraction))
        
        splitting_directory = "-".join(splitting_directory_parts)
    
    else:
        splitting_directory = "no_split"
    
    # Preprocessing directory
    
    preprocessing_directory_parts = []
    
    if data_set.map_features:
        preprocessing_directory_parts.append("features_mapped")
    
    if data_set.feature_selection:
        feature_selection_part = normaliseString(data_set.feature_selection)
        if data_set.feature_selection_parameters:
            for parameter in data_set.feature_selection_parameters:
                feature_selection_part += "_" + normaliseString(str(
                    parameter))
        preprocessing_directory_parts.append(feature_selection_part)
    
    if data_set.example_filter:
        example_filter_part = normaliseString(data_set.example_filter)
        if data_set.example_filter_parameters:
            for parameter in data_set.example_filter_parameters:
                example_filter_part += "_" + normaliseString(str(
                    parameter))
        preprocessing_directory_parts.append(example_filter_part)
    
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
        preprocessing_directory = "no_preprocessing"
    
    # Complete path
    
    directory = os.path.join(data_set_directory, splitting_directory,
        preprocessing_directory)
    
    return directory
