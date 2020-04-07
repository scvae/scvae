# ======================================================================== #
#
# Copyright (c) 2017 - 2019 scVAE authors
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

import numpy
import pandas

from scvae.utilities import normalise_string

EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES = 25
EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES_PER_CLASS = 3


def standard_deviation(a, axis=None, ddof=0, batch_size=None):
    if (not isinstance(a, numpy.ndarray) or axis is not None
            or batch_size is None):
        return a.std(axis=axis, ddof=ddof)
    return numpy.sqrt(variance(
        a=a,
        axis=axis,
        ddof=ddof,
        batch_size=batch_size
    ))


def variance(a, axis=None, ddof=0, batch_size=None):

    if (not isinstance(a, numpy.ndarray) or axis is not None
            or batch_size is None):
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


def build_directory_path(base_directory, data_set, splitting_method=None,
                         splitting_fraction=None, preprocessing=True):

    data_set_directory = os.path.join(base_directory, data_set.name)

    # Splitting directory

    if splitting_method:

        splitting_directory_parts = ["split"]

        if splitting_method == "default":
            splitting_method = data_set.default_splitting_method

        if (splitting_method == "indices" and len(data_set.split_indices) == 3
                or not splitting_fraction):
            splitting_directory_parts.append(splitting_method)
        else:
            splitting_directory_parts.append(
                "{}_{}".format(splitting_method, splitting_fraction)
            )

        splitting_directory = "-".join(splitting_directory_parts)

    else:
        splitting_directory = "no_split"

    # Preprocessing directory

    preprocessing_directory_parts = []

    if data_set.features_mapped:
        preprocessing_directory_parts.append("features_mapped")

    if data_set.feature_selection_method:
        feature_selection_part = normalise_string(
            data_set.feature_selection_method)
        if data_set.feature_selection_parameters:
            for parameter in data_set.feature_selection_parameters:
                feature_selection_part += "_" + normalise_string(str(
                    parameter))
        preprocessing_directory_parts.append(feature_selection_part)

    if data_set.example_filter_method:
        example_filter_part = normalise_string(data_set.example_filter_method)
        if data_set.example_filter_parameters:
            for parameter in data_set.example_filter_parameters:
                example_filter_part += "_" + normalise_string(str(
                    parameter))
        preprocessing_directory_parts.append(example_filter_part)

    if preprocessing and data_set.preprocessing_methods:
        preprocessing_directory_parts.extend(
            map(normalise_string, data_set.preprocessing_methods)
        )

    if preprocessing and data_set.noisy_preprocessing_methods:
        preprocessing_directory_parts.append("noisy")
        preprocessing_directory_parts.extend(
            map(normalise_string, data_set.noisy_preprocessing_methods)
        )

    if preprocessing_directory_parts:
        preprocessing_directory = "-".join(preprocessing_directory_parts)
    else:
        preprocessing_directory = "no_preprocessing"

    # Complete path

    directory = os.path.join(
        data_set_directory,
        splitting_directory,
        preprocessing_directory
    )

    return directory


def indices_for_evaluation_subset(evaluation_set,
                                  maximum_number_of_examples_per_class=None,
                                  total_maximum_number_of_examples=None):

    if maximum_number_of_examples_per_class is None:
        maximum_number_of_examples_per_class = (
            EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES_PER_CLASS)

    if total_maximum_number_of_examples is None:
        total_maximum_number_of_examples = (
            EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES)

    random_state = numpy.random.RandomState(80)

    if evaluation_set.has_labels:

        if evaluation_set.label_superset:
            class_names = evaluation_set.superset_class_names
            labels = evaluation_set.superset_labels
        else:
            class_names = evaluation_set.class_names
            labels = evaluation_set.labels

        subset = set()

        for class_name in class_names:
            class_label_indices = numpy.argwhere(labels == class_name)
            random_state.shuffle(class_label_indices)
            subset.update(
                *class_label_indices[:maximum_number_of_examples_per_class])

    else:
        subset = numpy.random.permutation(evaluation_set.number_of_examples)[
            :total_maximum_number_of_examples]
        subset = set(subset)

    return subset


def save_values(values, name, row_names=None, column_names=None,
                directory=None):

    safe_name = "-".join([normalise_string(part) for part in name.split("-")])
    filename = "{}.tsv.gz".format(safe_name)
    path = os.path.join(directory, filename)

    table = pandas.DataFrame(
        data=values, index=row_names, columns=column_names)

    if not os.path.exists(directory):
        os.makedirs(directory)

    table.to_csv(path, sep="\t")
