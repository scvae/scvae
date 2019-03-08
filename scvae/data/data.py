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
import shutil
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
from data import loading, parsing, processing
from data.internal_io import load_data_dictionary, save_data_dictionary
from data.sparse import SparseRowMatrix
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

default_tags = {
    "example": "example",
    "feature": "feature",
    "mapped feature": "mapped feature",
    "class": "class",
    "type": "value",
    "item": "item"
}

default_excluded_classes = ["No class"]

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
        self.name, data_set_dictionary = parsing.parse_input(
            input_file_or_name)
        
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
            if os.path.exists(self.directory):
                shutil.rmtree(self.directory)
            parsing.save_data_set_dictionary_as_json_file(
                data_set_dictionary,
                self.name,
                self.directory
            )
        
        # Find data set
        self.title, self.specifications = parsing.find_data_set(
            self.name, directory)
        
        # Tags (with names for examples, feature, and values) of data set
        self.tags = postprocessTags(self.specifications.get(
            "tags", default_tags))
        
        # Example type for data set
        self.example_type = self.specifications.get("example type", "unknown")
        
        # Maximum value of data set
        self.maximum_value = self.specifications.get("maximum value")
        
        # Discreteness
        self.discreteness = self.example_type == "counts" \
            or (self.maximum_value != None and self.maximum_value == 255)
        
        # Feature dimensions for data set
        self.feature_dimensions = self.specifications.get(
            "feature dimensions")
        
        # Literature probabilities for data set
        self.literature_probabilities = None
        
        # Label super set for data set
        self.label_superset = self.specifications.get("label superset")
        self.superset_labels = None
        self.number_of_superset_classes = None
        
        # Label palette for data set
        self.class_palette = self.specifications.get("class palette")
        self.superset_class_palette = supersetClassPaletteBuilder(
            self.class_palette, self.label_superset)
        
        # Excluded classes for data set
        self.excluded_classes = self.specifications.get("excluded classes")
        
        # Excluded classes for data set
        self.excluded_superset_classes = self.specifications.get(
            "excluded superset classes")
        
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
        sorted_class_names = self.specifications.get("sorted class names")
        self.label_sorter = createLabelSorter(sorted_class_names)
        sorted_superset_class_names = self.specifications.get(
            "sorted superset class names")
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
            data_set_preprocessing_methods = self.specifications.get(
                "preprocessing methods")
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
        
        # PCA limits for data set
        self.pca_limits = None
        
        # Noisy preprocessing
        self.noisy_preprocessing_methods = noisy_preprocessing_methods
        
        if self.preprocessed:
            self.noisy_preprocessing_methods = []
        
        if self.noisy_preprocessing_methods:
            self.noisy_preprocess = processing.build_preprocessor(
                self.noisy_preprocessing_methods,
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
            
            if self.class_palette is None:
                self.class_palette = classPaletteBuilder(self.class_names)
            
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
                
                if self.superset_class_palette is None:
                    self.superset_class_palette = supersetClassPaletteBuilder(
                        self.class_names, self.label_superset)
        
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
            data_dictionary = load_data_dictionary(sparse_path)
            print()
        else:
            URLs = self.specifications.get("URLs", None)
            original_paths = loading.acquire_data_set(self.title, URLs,
                self.original_directory)
            data_format = self.specifications.get("format")
            
            loading_time_start = time()
            data_dictionary = loading.load_original_data_set(original_paths,
                data_format)
            loading_duration = time() - loading_time_start
            
            print()
            
            if loading_duration > maximum_duration_before_saving:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving data set.")
                save_data_dictionary(data_dictionary, sparse_path)
                
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
            data_dictionary = load_data_dictionary(sparse_path)
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
                
                values, feature_names = processing.map_features(
                    values, feature_names, self.feature_mapping)
                
                self.features_mapped = True
                self.tags = updateTagForMappedFeatures(self.tags)
                
                duration = time() - start_time
                print("Features mapped ({}).".format(formatDuration(duration)))
                
                print()
            
            if not self.preprocessed and self.preprocessing_methods:
                
                print("Preprocessing values.")
                start_time = time()
                
                preprocessing_function = processing.build_preprocessor(
                    self.preprocessing_methods)
                preprocessed_values = preprocessing_function(values)
                
                duration = time() - start_time
                print("Values preprocessed ({}).".format(formatDuration(duration)))
                
                print()
            
            else:
                preprocessed_values = None
            
            if self.feature_selection:
                values_dictionary, feature_names = processing.select_features(
                    {"original": values,
                     "preprocessed": preprocessed_values},
                    self.feature_names,
                    self.feature_selection,
                    self.feature_selection_parameters
                )
                
                values = values_dictionary["original"]
                preprocessed_values = values_dictionary["preprocessed"]
            
                print()
                
            if self.example_filter:
                values_dictionary, example_names, labels = processing.filter_examples(
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
                save_data_dictionary(data_dictionary, sparse_path)
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
            data_dictionary = load_data_dictionary(sparse_path)
        
        else:
            
            binarising_time_start = time()
            
            if self.preprocessing_methods != binarise_preprocessing:
                
                print("Binarising values.")
                start_time = time()
                
                binarisation_function = processing.build_preprocessor(
                    binarise_preprocessing)
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
                save_data_dictionary(data_dictionary, sparse_path)
        
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
            split_data_dictionary = load_data_dictionary(sparse_path)
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
            split_data_dictionary = processing.split_data_set(
                data_dictionary, method, fraction)
            splitting_duration = time() - splitting_time_start
            
            print()
            
            if splitting_duration > maximum_duration_before_saving:
                if not os.path.exists(self.preprocess_directory):
                    os.makedirs(self.preprocess_directory)
                
                print("Saving split data sets.")
                save_data_dictionary(split_data_dictionary, sparse_path)
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

def postprocessTags(tags):
    if "item" in tags and tags["item"]:
        value_tag = tags["item"] + " " + tags["type"]
    else:
        value_tag = tags["type"]
    tags["value"] = value_tag
    return tags

def classPaletteBuilder(class_names):

    brewer_palette = seaborn.color_palette("Set3")

    if len(class_names) <= len(brewer_palette):
        class_palette = {
            c: brewer_palette[i] for i, c in enumerate(class_names)
        }
    else:
        class_palette = None

    return class_palette

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

def defaultFeatureParameters(feature_selection = None,
    number_of_features = None):
    
    if feature_selection:
        feature_selection = normaliseString(feature_selection)
        M = number_of_features
        
        if feature_selection == "remove_zeros":
            feature_selection_parameters = None
        
        elif feature_selection == "keep_variances_above":
            feature_selection_parameters = [0.5]
        
        elif feature_selection == "keep_highest_variances":
            feature_selection_parameters = [int(M/2)]
        
        else:
            feature_selection_parameters = None
    
    else:
        feature_selection_parameters = None
    
    return feature_selection_parameters

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

def supersetClassPaletteBuilder(class_palette, label_superset):
    
    if (class_palette is None or label_superset is None
            or label_superset == "infer"):
        superset_class_palette = None
    
    else:
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

def createLabelSorter(sorted_class_names = None):
    
    if not sorted_class_names:
        sorted_class_names = []
    
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
