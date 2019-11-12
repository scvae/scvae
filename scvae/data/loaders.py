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

import gzip
import os
import pickle
import struct
import tarfile

import loompy
import numpy
import pandas
import scipy
import tables

from scvae.utilities import normalise_string

# List name strings are normalised, so no need to check for
# capitalisation or spacing variants
LIST_NAME_GUESSES = {
    "example": [
        "barcodes", "cells", "cell_names", "cell_ids"
        "samples", "sample_names", "sample_ids",
        "examples", "example_names", "example_ids"
    ],
    "feature": [
        "genes", "gene_names", "gene_ids",
        "features", "feature_names", "feature_ids"
    ]
}

LOADERS = {}


def _register_loader(name):
    def decorator(function):
        LOADERS[name] = function
        return function
    return decorator


@_register_loader("macosko")
def _load_macokso_data_set(paths):

    values, column_headers, row_indices = _load_tab_separated_matrix(
        paths["values"]["full"], numpy.float32)

    values = values.T
    example_names = numpy.array(column_headers)

    feature_column = 0
    feature_names = numpy.array(row_indices)[:, feature_column]

    labels = None
    labels_paths = paths.get("labels", {})
    full_labels_path = labels_paths.get("full")
    if full_labels_path:
        labels = _load_labels_from_delimiter_separeted_values(
            path=paths["labels"]["full"],
            label_column=1,
            example_column=0,
            example_names=example_names,
            header=None,
            default_label=0
        )

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }

    return data_dictionary


@_register_loader("10x")
def _load_10x_data_set(paths):

    data_dictionary = _load_values_from_10x_data_set(paths["values"]["full"])
    values = data_dictionary["values"]
    example_names = data_dictionary["example names"]
    feature_names = data_dictionary["feature names"]

    labels = None
    labels_paths = paths.get("labels", {})
    full_labels_path = labels_paths.get("full")
    if full_labels_path:
        labels = _load_labels_from_delimiter_separeted_values(
            path=full_labels_path,
            label_column="celltype",
            example_column="barcodes",
            example_names=example_names,
            dtype="U"
        )

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }

    return data_dictionary


@_register_loader("h5")
def _load_h5_data_set(paths):

    data_dictionary = _load_sparse_matrix_in_hdf5_format(
        paths["values"]["full"])
    values = data_dictionary["values"]
    example_names = data_dictionary["example names"]
    feature_names = data_dictionary["feature names"]

    labels = None
    labels_paths = paths.get("labels", {})
    full_labels_path = labels_paths.get("full")
    if full_labels_path:
        labels = _load_labels_from_delimiter_separeted_values(
            path=full_labels_path,
            example_names=example_names,
            dtype="U"
        )

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names
    }

    return data_dictionary


@_register_loader("10x_combine")
def _load_and_combine_10x_data_sets(paths):

    # Initialisation

    value_sets = {}
    example_name_sets = {}
    feature_name_sets = {}
    genome_names = {}

    # Loading values from separate data sets

    for class_name, path in paths["all"].items():
        data_dictionary = _load_values_from_10x_data_set(path)
        value_sets[class_name] = data_dictionary["values"]
        example_name_sets[class_name] = data_dictionary["example names"]
        feature_name_sets[class_name] = data_dictionary["feature names"]
        genome_names[class_name] = data_dictionary["genome name"]

    # Check for multiple genomes

    class_name, genome_name = genome_names.popitem()

    for other_class_name, other_genome_name in genome_names.items():
        if not genome_name == other_genome_name:
            raise ValueError(
                "The genome names for \"{}\" and \"{}\" do not match."
                .format(class_name, other_class_name)
            )

    # Infer labels

    label_sets = {}

    for class_name in example_name_sets:
        label_sets[class_name] = numpy.array(
            [class_name] * example_name_sets[class_name].shape[0]
        )

    # Combine data sets

    def sort_values(d):
        return [v for k, v in sorted(d.items())]

    values = scipy.sparse.vstack(sort_values(value_sets))
    example_names = numpy.concatenate(sort_values(example_name_sets))
    labels = numpy.concatenate(sort_values(label_sets))

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


@_register_loader("tcga")
def _load_tcga_data_set(paths):

    # Values, example names, and feature names

    values, column_headers, row_indices = _load_tab_separated_matrix(
        paths["values"]["full"], numpy.float32)

    values = values.T
    values = numpy.power(2, values) - 1
    values = numpy.round(values)

    example_names = numpy.array(column_headers)

    feature_id_column = 0
    feature_ids = numpy.array(row_indices)[:, feature_id_column]

    # Labels

    labels = None
    labels_paths = paths.get("labels", {})
    full_labels_path = labels_paths.get("full")
    if full_labels_path:
        labels = _load_labels_from_delimiter_separeted_values(
            path=paths["labels"]["full"],
            label_column="_primary_site",
            example_column="sampleID",
            example_names=example_names,
            dtype="U",
            default_label="No class"
        )

    # Feature mapping

    feature_mapping = dict()
    path = paths["feature mapping"]["full"]

    with gzip.open(path, mode="rt") as feature_mapping_file:

        for row in feature_mapping_file:
            if row.startswith("#"):
                continue
            row_elements = row.split()
            feature_name = row_elements[1]
            feature_id = row_elements[0]
            if feature_name not in feature_mapping:
                feature_mapping[feature_name] = []
            feature_mapping[feature_name].append(feature_id)

    # Dictionary

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_ids,
        "feature mapping": feature_mapping
    }

    return data_dictionary


@_register_loader("gtex")
def _load_gtex_data_set(paths):

    # Values, example names and feature names

    values, column_headers, row_indices = _load_tab_separated_matrix(
        paths["values"]["full"], numpy.float32)

    values = values.T

    example_names = numpy.array(column_headers)

    feature_id_column = 0
    feature_name_column = 1

    feature_ids = numpy.array(row_indices)[:, feature_id_column]
    feature_names = numpy.array(row_indices)[:, feature_name_column]

    # Labels

    labels = None
    labels_paths = paths.get("labels", {})
    full_labels_path = labels_paths.get("full")
    if full_labels_path:
        labels = _load_labels_from_delimiter_separeted_values(
            path=paths["labels"]["full"],
            label_column="SMTSD",
            example_column="SAMPID",
            example_names=example_names,
            dtype="U"
        )

    # Feature mapping

    feature_mapping = dict()

    for feature_name, feature_id in zip(feature_names, feature_ids):
        if feature_name not in feature_mapping:
            feature_mapping[feature_name] = []
        feature_mapping[feature_name].append(feature_id)

    # Dictionary

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_ids,
        "feature mapping": feature_mapping
    }

    return data_dictionary


@_register_loader("loom")
def _load_loom_data_set(paths):

    values = labels = example_names = feature_names = batch_indices = None

    with loompy.connect(paths["all"]["full"]) as data_file:

        values = data_file[:, :].T
        n_examples, n_features = values.shape

        if "ClusterName" in data_file.ca:
            labels = data_file.ca["ClusterName"].flatten()
        elif "ClusterID" in data_file.ca:
            cluster_ids = data_file.ca["ClusterID"].flatten()
            if "CellTypes" in data_file.attrs:
                class_names = numpy.array(data_file.attrs["CellTypes"])
                class_name_from_class_id = numpy.vectorize(
                    lambda class_id: class_names[int(class_id)]
                )
                labels = class_name_from_class_id(cluster_ids)
            else:
                labels = cluster_ids

        if "CellID" in data_file.ca:
            example_names = data_file.ca["CellID"].flatten().astype("U")
        elif "Cell" in data_file.ca:
            example_names = data_file.ca["Cell"].flatten()
        else:
            example_names = numpy.array([
                "Cell {}".format(j + 1) for j in range(n_examples)])

        if "Gene" in data_file.ra:
            feature_names = data_file.ra["Gene"].flatten().astype("U")
        else:
            feature_names = numpy.array([
                "Gene {}".format(j + 1) for j in range(n_features)])

        if "BatchID" in data_file.ca:
            batch_indices = data_file.ca["BatchID"].flatten()

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
        "batch indices": batch_indices
    }

    return data_dictionary


@_register_loader("matrix_fbe")
def _load_fbe_matrix_as_data_set(paths):
    return _load_values_and_labels_from_matrix(
        paths=paths,
        orientation="fbe"
    )


@_register_loader("matrix_ebf")
def _load_ebf_matrix_as_data_set(paths):
    return _load_values_and_labels_from_matrix(
        paths=paths,
        orientation="ebf"
    )


@_register_loader("mnist_original")
def _load_original_mnist_data_set(paths):

    values = {}

    for kind in paths["values"]:
        with gzip.open(paths["values"][kind], mode="rb") as values_stream:
            _, m, r, c = struct.unpack(">IIII", values_stream.read(16))
            values_buffer = values_stream.read(m * r * c)
            values_flat = numpy.frombuffer(values_buffer, dtype=numpy.uint8)
            values[kind] = values_flat.reshape(-1, r * c)

    n = r * c

    labels = {}

    for kind in paths["labels"]:
        with gzip.open(paths["labels"][kind], mode="rb") as labels_stream:
            _, m = struct.unpack(">II", labels_stream.read(8))
            labels_buffer = labels_stream.read(m)
            labels[kind] = numpy.frombuffer(labels_buffer, dtype=numpy.int8)

    m_training = values["training"].shape[0]
    m_test = values["test"].shape[0]
    m = m_training + m_test

    split_indices = {
        "training": slice(0, m_training),
        "test": slice(m_training, m)
    }

    values = numpy.concatenate((values["training"], values["test"]))
    labels = numpy.concatenate((labels["training"], labels["test"]))

    values = values.astype(numpy.float32)

    example_names = numpy.array(["image {}".format(i + 1) for i in range(m)])
    feature_names = numpy.array(["pixel {}".format(j + 1) for j in range(n)])

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }

    return data_dictionary


@_register_loader("mnist_normalised")
def _load_normalised_mnist_data_set(paths):

    with gzip.open(paths["all"]["full"], mode="r") as data_file:
        ((values_training, labels_training),
         (values_validation, labels_validation),
         (values_test, labels_test)) = pickle.load(
            data_file, encoding="latin1")

    m_training = values_training.shape[0]
    m_validation = values_validation.shape[0]
    m_training_validation = m_training + m_validation
    m_test = values_test.shape[0]
    m = m_training_validation + m_test

    split_indices = {
        "training": slice(0, m_training),
        "validation": slice(m_training, m_training_validation),
        "test": slice(m_training_validation, m)
    }

    values = numpy.concatenate((
        values_training, values_validation, values_test
    ))

    labels = numpy.concatenate((
        labels_training, labels_validation, labels_test
    ))

    n = values.shape[1]

    example_names = numpy.array(["image {}".format(i + 1) for i in range(m)])
    feature_names = numpy.array(["pixel {}".format(j + 1) for j in range(n)])

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }

    return data_dictionary


@_register_loader("mnist_binarised")
def _load_binarised_mnist_data_set(paths):

    values = {}

    for kind in paths["values"]:
        values[kind] = numpy.loadtxt(paths["values"][kind], numpy.float32)

    m_training = values["training"].shape[0]
    m_validation = values["validation"].shape[0]
    m_training_validation = m_training + m_validation
    m_test = values["test"].shape[0]
    m = m_training_validation + m_test

    split_indices = {
        "training": slice(0, m_training),
        "validation": slice(m_training, m_training_validation),
        "test": slice(m_training_validation, m)
    }

    values = numpy.concatenate((
        values["training"], values["validation"], values["test"]
    ))

    n = values.shape[1]

    example_names = numpy.array(["image {}".format(i + 1) for i in range(m)])
    feature_names = numpy.array(["pixel {}".format(j + 1) for j in range(n)])

    data_dictionary = {
        "values": values,
        "labels": None,
        "example names": example_names,
        "feature names": feature_names,
        "split indices": split_indices
    }

    return data_dictionary


@_register_loader("development")
def _load_development_data_set(**kwargs):
    return _create_development_data_set(
        n_examples=10000,
        n_features=25,
        scale=10,
        update_probability=0.0001
    )


def _load_values_and_labels_from_matrix(paths, orientation=None):

    # Values

    values, column_headers, row_indices = _load_tab_separated_matrix(
        paths["values"]["full"], numpy.float32)

    if orientation == "fbe":
        values = values.T
        example_names = column_headers
        feature_names = row_indices
    elif orientation == "ebf":
        example_names = row_indices
        feature_names = column_headers
    elif orientation is None:
        raise ValueError(" ".join[
            "Orientation of matrix not set."
            "`fbe`: rows as features; columns as examples."
            "`ebf`: rows as examples; columns as features."
        ])
    else:
        raise ValueError("`{}` not a valid orientation.".format(orientation))

    n_examples, n_features = values.shape

    if example_names is None:
        example_names = ["example {}".format(i + 1) for i in range(n_examples)]

    if feature_names is None:
        feature_names = ["feature {}".format(j + 1) for j in range(n_features)]

    example_names = numpy.array(example_names).flatten()
    feature_names = numpy.array(feature_names).flatten()

    # Labels

    if "labels" in paths:
        labels = _load_labels_from_delimiter_separeted_values(
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


def _load_values_from_10x_data_set(path):

    parent_paths = set()

    multiple_directories_error = NotImplementedError(
        "Cannot handle 10x data sets with multiple directories."
    )

    if path.endswith(".h5"):
        with tables.open_file(path, mode="r") as f:

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
        with tarfile.open(path, mode="r:gz") as tarball:
            for member in sorted(tarball, key=lambda member: member.name):
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


def _load_sparse_matrix_in_hdf5_format(path, example_names_key=None,
                                       feature_names_key=None):

    parent_paths = set()
    table = {}

    with tables.open_file(path, mode="r") as f:

        for node in f.walk_nodes(where="/", classname="Array"):
            node_path = node._v_pathname
            parent_path, node_name = os.path.split(node_path)
            parent_paths.add(parent_path)
            if len(parent_paths) > 1:
                raise NotImplementedError(
                    "Cannot handle HDF5 data sets with multiple directories.")
            table[node.name] = node.read()

    values = scipy.sparse.csc_matrix(
        (table["data"], table["indices"], table["indptr"]),
        shape=table["shape"]
    )
    table.pop("data")
    table.pop("indices")
    table.pop("indptr")
    table.pop("shape")

    n_examples, n_features = values.shape
    n = {
        "example": n_examples,
        "feature": n_features
    }

    def _find_list_of_names(list_name_guesses, kind):
        if list_name_guesses is None:
            list_name_guesses = LIST_NAME_GUESSES[kind]
        elif not isinstance(list_name_guesses, list):
            list_name_guesses = [list_name_guesses]
        list_of_names = None
        for list_name_guess in list_name_guesses:
            for table_key in table:
                if list_name_guess == normalise_string(table_key):
                    list_of_names = table[table_key]
            if list_of_names is not None:
                break
        list_of_names = numpy.array(
            ["{} {}".format(kind, i + 1) for i in range(n[kind])])

    example_names = _find_list_of_names(example_names_key, kind="example")
    feature_names = _find_list_of_names(feature_names_key, kind="feature")

    data_dictionary = {
        "values": values,
        "example names": example_names,
        "feature names": feature_names
    }

    return data_dictionary


def _load_tab_separated_matrix(tsv_path, data_type=None):

    tsv_extension = tsv_path.split(os.extsep, 1)[-1]

    if tsv_extension == "tsv":
        open_file = open
    elif tsv_extension.endswith("gz"):
        open_file = gzip.open
    else:
        raise NotImplementedError(
            "Loading from file with extension `{}` not implemented.".format(
                tsv_extension)
        )

    values = []
    row_indices = []
    column_headers = None

    with open_file(tsv_path, mode="rt") as tsv_file:

        while not column_headers:

            row_elements = next(tsv_file).split()

            # Skip, if row could not be split into elements
            if len(row_elements) <= 1:
                continue

            # Skip, if row only contains two integers before header
            # (assumed to be the shape of the matrix)
            elif (len(row_elements) == 2
                    and all([element.isdigit() for element in row_elements])):
                continue

            elif all(_is_float(element) for element in row_elements):
                break

            column_headers = row_elements

        if column_headers:
            row_elements = next(tsv_file).split()

        for i, element in enumerate(row_elements):
            if _is_float(element):
                column_offset = i
                break

        if column_headers:
            column_header_offset = column_offset - (
                len(row_elements) - len(column_headers)
            )
            column_headers = column_headers[column_header_offset:]

        def parse_row_elements(row_elements):
            row_index = row_elements[:column_offset]
            if row_index:
                row_indices.append(row_index)
            row_values = list(map(float, row_elements[column_offset:]))
            values.append(row_values)

        parse_row_elements(row_elements)

        for row in tsv_file:
            parse_row_elements(row.split())

    values = numpy.array(values, data_type)

    if not row_indices:
        row_indices = None

    return values, column_headers, row_indices


def _load_labels_from_delimiter_separeted_values(
        path, label_column=1, example_column=0,
        example_names=None, delimiter=None,
        header="infer", dtype=None, default_label=None):

    if not delimiter:
        if path.endswith(".csv"):
            delimiter = ","
        else:
            delimiter = "\t"

    if path.endswith(".gz"):
        open_file = gzip.open
    else:
        open_file = open

    with open_file(path, mode="rt") as labels_file:
        first_row_elements = next(labels_file).split()
        second_row_elements = next(labels_file).split()

    if len(first_row_elements) == 1 and len(second_row_elements) == 1:
        label_column = 0
        index_column = None
        use_columns = None
        example_names = None
        if header == "infer":
            header = None
    else:
        index_column = example_column
        use_columns = [example_column, label_column]
        if isinstance(label_column, int):
            label_column -= 1

    metadata = pandas.read_csv(
        path,
        index_col=index_column,
        usecols=use_columns,
        delimiter=delimiter,
        header=header
    )

    if isinstance(label_column, int):
        label_column = metadata.columns[label_column]

    unordered_labels = metadata[label_column]

    if example_names is not None:

        labels = numpy.zeros(example_names.shape, unordered_labels.dtype)

        for example_name, label in unordered_labels.items():
            labels[example_names == example_name] = label

        if default_label:
            labels[labels == 0] = default_label

    else:
        labels = unordered_labels.values

    if dtype is None and labels.dtype == "object":
        dtype = "U"

    if dtype:
        labels = labels.astype(dtype)

    return labels


def _create_development_data_set(
        n_examples=10000, n_features=25, scale=10, update_probability=0.0001):

    random_state = numpy.random.RandomState(60)

    values = numpy.empty((n_examples, n_features), numpy.float32)
    labels = numpy.empty(n_examples, numpy.int32)

    r = numpy.empty((n_examples, n_features))
    p = numpy.empty((n_examples, n_features))
    dropout = numpy.empty((n_examples, n_features))

    def r_draw():
        return scale * random_state.rand(n_features)

    def p_draw():
        return random_state.rand(n_features)

    def dropout_draw():
        return random_state.rand(n_features)

    r_type = r_draw()
    p_type = p_draw()
    dropout_type = dropout_draw()

    label = 1

    for i in range(n_examples):
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

    shuffled_indices = random_state.permutation(n_examples)

    r = r[shuffled_indices]
    p = p[shuffled_indices]
    dropout = dropout[shuffled_indices]
    labels = labels[shuffled_indices]

    no_class_indices = random_state.permutation(n_examples)[
        :int(0.1 * n_examples)]
    labels[no_class_indices] = 0

    labels = labels.astype(str)

    for i in range(n_examples):
        for j in range(n_features):
            value = random_state.negative_binomial(r[i, j], p[i, j])
            value_dropout = random_state.binomial(1, dropout[i, j])
            values[i, j] = value_dropout * value

    example_names = numpy.array(
        ["example {}".format(i + 1) for i in range(n_examples)])
    feature_ids = numpy.array(
        ["feature {}".format(j + 1) for j in range(n_features)])

    feature_names = ["feature " + n for n in "ABCDE"]
    feature_id_groups = numpy.split(feature_ids, len(feature_names))

    feature_mapping = {
        feature_name: feature_id_group.tolist()
        for feature_name, feature_id_group in
        zip(feature_names, feature_id_groups)
    }

    data_dictionary = {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_ids,
        "feature mapping": feature_mapping
    }

    return data_dictionary


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
