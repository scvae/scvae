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

import os
from time import time

import numpy
import scipy
import tables

from scvae.utilities import normalise_string, format_duration


def load_data_dictionary(path):

    def load(tables_file, group=None):

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
                        tables_file, group=node)
                elif node_title.endswith("values"):
                    data_dictionary[node_title] = _load_sparse_matrix(
                        tables_file, group=node)
                elif node_title == "split indices":
                    data_dictionary[node_title] = _load_split_indices(
                        tables_file, group=node)
                elif node_title == "feature mapping":
                    data_dictionary[node_title] = _load_feature_mapping(
                        tables_file, group=node)
                else:
                    raise NotImplementedError(
                        "Loading group `{}` not implemented.".format(
                            node_title)
                    )
            elif isinstance(node, tables.Array):
                data_dictionary[node_title] = _load_array_or_other_type(node)
            else:
                raise NotImplementedError(
                    "Loading node `{}` not implemented.".format(node_title)
                )

        return data_dictionary

    start_time = time()

    with tables.open_file(path, "r") as tables_file:
        data_dictionary = load(tables_file)

    duration = time() - start_time
    print("Data loaded ({}).".format(format_duration(duration)))

    return data_dictionary


def save_data_dictionary(data_dictionary, path):

    directory, filename = os.path.split(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    def save(data_dictionary, tables_file, group_title=None):

        if group_title:
            group = tables_file.create_group(
                "/", normalise_string(group_title), group_title)
        else:
            group = tables_file.root

        for title, value in data_dictionary.items():

            if isinstance(value, scipy.sparse.csr_matrix):
                _save_sparse_matrix(value, title, group, tables_file)
            elif isinstance(value, (numpy.ndarray, list)):
                _save_array(value, title, group, tables_file)
            elif title == "split indices":
                _save_split_indices(value, title, group, tables_file)
            elif title == "feature mapping":
                _save_feature_mapping(value, title, group, tables_file)
            elif value is None:
                _save_string(str(value), title, group, tables_file)
            elif title.endswith("set"):
                save(value, tables_file, group_title=title)
            else:
                raise NotImplementedError(
                    "Saving type {} for title \"{}\" has not been implemented."
                    .format(type(value), title)
                )

    start_time = time()

    filters = tables.Filters(complib="zlib", complevel=5)

    with tables.open_file(path, "w", filters=filters) as tables_file:
        save(data_dictionary, tables_file)

    duration = time() - start_time
    print("Data saved ({}).".format(format_duration(duration)))


def _load_array_or_other_type(node):

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


def _load_sparse_matrix(tables_file, group):

    arrays = {}

    for array in tables_file.iter_nodes(group, "Array"):
        arrays[array.title] = array.read()

    sparse_matrix = scipy.sparse.csr_matrix(
        (arrays["data"], arrays["indices"], arrays["indptr"]),
        shape=arrays["shape"]
    )

    return sparse_matrix


def _load_split_indices(tables_file, group):

    split_indices = {}

    for array in tables_file.iter_nodes(group, "Array"):
        start, stop = array.read()
        split_indices[array.title] = slice(start, stop)

    return split_indices


def _load_feature_mapping(tables_file, group):

    feature_lists = {}

    for array in tables_file.iter_nodes(group, "Array"):
        feature_lists[array.title] = array.read().tolist()

    feature_names = feature_lists["feature_names"]
    feature_counts = feature_lists["feature_counts"]
    feature_ids = feature_lists["feature_ids"]

    feature_mapping = {}

    for feature_name, feature_count in zip(feature_names, feature_counts):
        feature_name = feature_name.decode("UTF-8")
        feature_id_set = [
            feature_ids.pop(0).decode("UTF-8") for i in range(feature_count)
        ]
        feature_mapping[feature_name] = feature_id_set

    return feature_mapping


def _save_array(array, title, group, tables_file):
    name = normalise_string(title)
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


def _save_string(string, title, group, tables_file):
    encoded_string = numpy.frombuffer(string.encode('UTF-8'), numpy.uint8)
    _save_array(encoded_string, title, group, tables_file)


def _save_sparse_matrix(sparse_matrix, title, group, tables_file):

    name = normalise_string(title)
    group = tables_file.create_group(group, name, title)

    for attribute in ("data", "indices", "indptr", "shape"):
        array = numpy.array(getattr(sparse_matrix, attribute))
        _save_array(array, attribute, group, tables_file)


def _save_split_indices(split_indices, title, group, tables_file):

    name = normalise_string(title)
    group = tables_file.create_group(group, name, title)

    for subset_name, subset_slice in split_indices.items():
        subset_slice_array = numpy.array(
            [subset_slice.start, subset_slice.stop])
        _save_array(subset_slice_array, subset_name, group, tables_file)


def _save_feature_mapping(feature_mapping, title, group, tables_file):

    name = normalise_string(title)
    group = tables_file.create_group(group, name, title)

    feature_names = []
    feature_counts = []
    feature_ids = []

    for feature_name, feature_id_set in feature_mapping.items():
        feature_names.append(feature_name)
        feature_counts.append(len(feature_id_set))
        feature_ids.extend(feature_id_set)

    feature_lists = {
        "feature_names": feature_names,
        "feature_counts": feature_counts,
        "feature_ids": feature_ids
    }

    for feature_list_name, feature_list in feature_lists.items():
        feature_list_array = numpy.array(feature_list)
        _save_array(feature_list_array, feature_list_name, group, tables_file)
