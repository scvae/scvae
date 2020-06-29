# ======================================================================== #
#
# Copyright (c) 2017 - 2020 scVAE authors
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

from functools import reduce
from time import time

import numpy
import scipy
import sklearn.preprocessing

from scvae.data.sparse import SparseRowMatrix
from scvae.defaults import defaults
from scvae.utilities import normalise_string, format_duration

PREPROCESSERS = {}


def map_features(values, feature_ids, feature_mapping):

    values = scipy.sparse.csc_matrix(values)

    n_examples, n_ids = values.shape
    n_features = len(feature_mapping)

    feature_name_from_id = {
        v: k for k, vs in feature_mapping.items() for v in vs
    }

    n_unknown_ids = 0

    for feature_id in feature_ids:
        if feature_id not in feature_name_from_id:
            feature_name_from_id[feature_id] = feature_id
            n_unknown_ids += 1

    if n_unknown_ids > 0:
        print(
            "{0} feature{1} cannot be mapped -- using original feature{1}."
            .format(n_unknown_ids, "s" if n_unknown_ids > 1 else "")
        )

    n_features += n_unknown_ids

    aggregated_values = numpy.zeros((n_examples, n_features), values.dtype)
    feature_names_with_index = dict()

    for i, feature_id in enumerate(feature_ids):

        feature_name = feature_name_from_id[feature_id]

        if feature_name in feature_names_with_index:
            index = feature_names_with_index[feature_name]
        else:
            index = len(feature_names_with_index)
            feature_names_with_index[feature_name] = index

        aggregated_values[:, index] += values[:, i].A.flatten()

    feature_names = list(feature_names_with_index.keys())

    feature_names_not_found = set(feature_mapping.keys()) - set(feature_names)
    n_feature_names_not_found = len(feature_names_not_found)
    n_features -= n_feature_names_not_found
    aggregated_values = aggregated_values[:, :n_features]

    if n_feature_names_not_found > 0:
        print(
            "Did not find any original features for {} new feature{}.".format(
                n_feature_names_not_found,
                "s" if n_feature_names_not_found > 1 else ""
            )
        )

    aggregated_values = SparseRowMatrix(aggregated_values)
    feature_names = numpy.array(feature_names)

    return aggregated_values, feature_names


def select_features(values_dictionary, feature_names, method=None,
                    parameters=None):

    method = normalise_string(method)

    print("Selecting features.")
    start_time = time()

    if type(values_dictionary) == dict:
        values = values_dictionary["original"]

    n_examples, n_features = values.shape

    if method == "remove_zeros":
        total_feature_sum = values.sum(axis=0)
        if isinstance(total_feature_sum, numpy.matrix):
            total_feature_sum = total_feature_sum.A.squeeze()
        indices = total_feature_sum != 0

    elif method == "keep_variances_above":
        variances = values.var(axis=0)
        if isinstance(variances, numpy.matrix):
            variances = variances.A.squeeze()
        if parameters:
            threshold = float(parameters[0])
        else:
            threshold = 0.5
        indices = variances > threshold

    elif method == "keep_highest_variances":
        variances = values.var(axis=0)
        if isinstance(variances, numpy.matrix):
            variances = variances.A.squeeze()
        variance_sorted_indices = numpy.argsort(variances)
        if parameters:
            number_to_keep = int(parameters[0])
        else:
            number_to_keep = int(n_examples/2)
        indices = numpy.sort(variance_sorted_indices[-number_to_keep:])

    else:
        raise ValueError(
            "Feature selection `{}` not found.".format(method))

    if method:
        error = Exception(
            "No features excluded using feature selection {}.".format(method))
        if indices.dtype == "bool" and all(indices):
            raise error
        elif indices.dtype != "bool" and len(indices) == n_features:
            raise error

    feature_selected_values = {}

    for version, values in values_dictionary.items():
        if values is not None:
            feature_selected_values[version] = values[:, indices]
        else:
            feature_selected_values[version] = None

    feature_selected_feature_names = feature_names[indices]

    n_features_changed = len(feature_selected_feature_names)

    duration = time() - start_time
    print("{} features selected, {} excluded ({}).".format(
        n_features_changed,
        n_features - n_features_changed,
        format_duration(duration)
    ))

    return feature_selected_values, feature_selected_feature_names


def filter_examples(values_dictionary, example_names,
                    method=None, parameters=None,
                    labels=None, excluded_classes=None,
                    superset_labels=None, excluded_superset_classes=None,
                    batch_indices=None, count_sum=None):

    print("Filtering examples.")
    start_time = time()

    method = normalise_string(method)

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

    n_examples, n_features = values.shape

    filter_indices = numpy.arange(n_examples)

    if method == "macosko":
        minimum_number_of_non_zero_elements = 900
        number_of_non_zero_elements = (values != 0).sum(axis=1)
        filter_indices = numpy.nonzero(
            number_of_non_zero_elements > minimum_number_of_non_zero_elements
        )[0]

    elif method == "inverse_macosko":
        maximum_number_of_non_zero_elements = 900
        number_of_non_zero_elements = (values != 0).sum(axis=1)
        filter_indices = numpy.nonzero(
            number_of_non_zero_elements <= maximum_number_of_non_zero_elements
        )[0]

    elif method in ["keep", "remove", "excluded_classes"]:

        if filter_labels is None:
            raise ValueError(
                "Cannot filter examples based on labels, "
                "since data set is unlabelled."
            )

        if method == "excluded_classes":
            method = "remove"
            parameters = filter_excluded_classes

        if method == "keep":
            label_indices = set()

            for parameter in parameters:
                for class_name in filter_class_names:

                    normalised_class_name = normalise_string(str(class_name))
                    normalised_parameter = normalise_string(str(parameter))

                    if normalised_class_name == normalised_parameter:
                        class_indices = filter_labels == class_name
                        label_indices.update(filter_indices[class_indices])

            filter_indices = filter_indices[list(label_indices)]

        elif method == "remove":

            for parameter in parameters:
                for class_name in filter_class_names:

                    normalised_class_name = normalise_string(str(class_name))
                    normalised_parameter = normalise_string(str(parameter))

                    if normalised_class_name == normalised_parameter:
                        label_indices = filter_labels != class_name
                        filter_labels = filter_labels[label_indices]
                        filter_indices = filter_indices[label_indices]

    elif method == "remove_count_sum_above":
        threshold = int(parameters[0])
        filter_indices = filter_indices[count_sum.reshape(-1) <= threshold]

    elif method == "random":
        n_samples = int(parameters[0])
        n_samples = min(n_samples, n_examples)
        random_state = numpy.random.RandomState(90)
        filter_indices = random_state.permutation(n_examples)[:n_samples]

    else:
        raise ValueError(
            "Example filter `{}` not found.".format(method))

    if method and len(filter_indices) == n_examples:
        raise Exception(
            "No examples filtered out using example filter `{}`."
            .format(method)
        )

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

    if batch_indices is not None:
        example_filtered_batch_indices = batch_indices[filter_indices]
    else:
        example_filtered_batch_indices = None

    n_examples_changed = len(example_filtered_example_names)

    duration = time() - start_time
    print("{} examples filtered out, {} remaining ({}).".format(
        n_examples - n_examples_changed,
        n_examples_changed,
        format_duration(duration)
    ))

    return (example_filtered_values, example_filtered_example_names,
            example_filtered_labels, example_filtered_batch_indices)


def build_preprocessor(preprocessing_methods, noisy=False):

    preprocessers = []

    for preprocessing_method in preprocessing_methods:

        if noisy and preprocessing_method == "binarise":
            preprocessing_method = "bernoulli_sample"

        preprocesser = PREPROCESSERS.get(preprocessing_method)

        if preprocesser is None:
            raise ValueError(
                "Preprocessing method `{}` not found."
                .format(preprocessing_method))

        preprocessers.append(preprocesser)

    if not preprocessing_methods:
        preprocessers.append(lambda x: x)

    def preprocess(values):
        return reduce(
            lambda v, p: p(v),
            preprocessers,
            values
        )

    return preprocess


def split_data_set(data_dictionary, method=None, fraction=None):

    if method is None:
        method = defaults["data"]["splitting_method"]
    if fraction is None:
        fraction = defaults["data"]["splitting_fraction"]

    print("Splitting data set.")
    start_time = time()

    if method == "default":
        if "split indices" in data_dictionary:
            method = "indices"
        else:
            method = "random"

    method = normalise_string(method)

    n = data_dictionary["values"].shape[0]

    random_state = numpy.random.RandomState(42)

    if method in ["random", "sequential"]:

        n_training_validation = int(fraction * n)
        n_training = int(fraction * n_training_validation)

        if method == "random":
            indices = random_state.permutation(n)
        else:
            indices = numpy.arange(n)

        training_indices = indices[:n_training]
        validation_indices = indices[n_training:n_training_validation]
        test_indices = indices[n_training_validation:]

    elif method == "indices":

        split_indices = data_dictionary["split indices"]

        training_indices = split_indices["training"]
        test_indices = split_indices["test"]

        if "validation" in split_indices:
            validation_indices = split_indices["validation"]
        else:
            n_training_validation = training_indices.stop
            n_all = test_indices.stop

            n_training = n_training_validation - (
                n_all - n_training_validation)

            training_indices = slice(n_training)
            validation_indices = slice(n_training, n_training_validation)

    elif method == "macosko":

        values = data_dictionary["values"]

        minimum_number_of_non_zero_elements = 900
        number_of_non_zero_elements = (values != 0).sum(axis=1)

        training_indices = numpy.nonzero(
            number_of_non_zero_elements > minimum_number_of_non_zero_elements
        )[0]

        test_validation_indices = numpy.nonzero(
            number_of_non_zero_elements <= minimum_number_of_non_zero_elements
        )[0]

        random_state.shuffle(test_validation_indices)

        n_validation_test = len(test_validation_indices)
        n_validation = int((1 - fraction) * n_validation_test)

        validation_indices = test_validation_indices[:n_validation]
        test_indices = test_validation_indices[n_validation:]

    else:
        raise ValueError("Splitting method `{}` not found.".format(method))

    split_data_dictionary = {
        "training set": {
            "values": data_dictionary["values"][training_indices],
            "preprocessed values": None,
            "binarised values": None,
            "labels": None,
            "example names":
                data_dictionary["example names"][training_indices],
            "batch indices": None
        },
        "validation set": {
            "values": data_dictionary["values"][validation_indices],
            "preprocessed values": None,
            "binarised values": None,
            "labels": None,
            "example names":
                data_dictionary["example names"][validation_indices],
            "batch indices": None
        },
        "test set": {
            "values": data_dictionary["values"][test_indices],
            "preprocessed values": None,
            "binarised values": None,
            "labels": None,
            "example names": data_dictionary["example names"][test_indices],
            "batch indices": None
        },
        "feature names": data_dictionary["feature names"],
        "class names": data_dictionary["class names"]
    }

    if "labels" in data_dictionary and data_dictionary["labels"] is not None:
        split_data_dictionary["training set"]["labels"] = (
            data_dictionary["labels"][training_indices])
        split_data_dictionary["validation set"]["labels"] = (
            data_dictionary["labels"][validation_indices])
        split_data_dictionary["test set"]["labels"] = (
            data_dictionary["labels"][test_indices])

    if ("preprocessed values" in data_dictionary
            and data_dictionary["preprocessed values"] is not None):
        split_data_dictionary["training set"]["preprocessed values"] = (
            data_dictionary["preprocessed values"][training_indices])
        split_data_dictionary["validation set"]["preprocessed values"] = (
            data_dictionary["preprocessed values"][validation_indices])
        split_data_dictionary["test set"]["preprocessed values"] = (
            data_dictionary["preprocessed values"][test_indices])

    if ("binarised values" in data_dictionary
            and data_dictionary["binarised values"] is not None):
        split_data_dictionary["training set"]["binarised values"] = (
            data_dictionary["binarised values"][training_indices])
        split_data_dictionary["validation set"]["binarised values"] = (
            data_dictionary["binarised values"][validation_indices])
        split_data_dictionary["test set"]["binarised values"] = (
            data_dictionary["binarised values"][test_indices])

    if ("batch indices" in data_dictionary
            and data_dictionary["batch indices"] is not None):
        split_data_dictionary["training set"]["batch indices"] = (
            data_dictionary["batch indices"][training_indices])
        split_data_dictionary["validation set"]["batch indices"] = (
            data_dictionary["batch indices"][validation_indices])
        split_data_dictionary["test set"]["batch indices"] = (
            data_dictionary["batch indices"][test_indices])

    duration = time() - start_time
    print("Data set split ({}).".format(format_duration(duration)))

    return split_data_dictionary


def _register_preprocessor(name):
    def decorator(function):
        PREPROCESSERS[name] = function
        return function
    return decorator


@_register_preprocessor("log")
def _log(values):
    return values.log1p()


@_register_preprocessor("exp")
def _exp(values):
    return values.expm1()


@_register_preprocessor("normalise")
def _normalise(values):
    return sklearn.preprocessing.normalize(values, norm="l2", axis=0)


@_register_preprocessor("binarise")
def _binarise(values):
    return sklearn.preprocessing.binarize(values, threshold=0.5)


@_register_preprocessor("bernoulli_sample")
def _bernoulli_sample(values):
    return numpy.random.binomial(1, values)
