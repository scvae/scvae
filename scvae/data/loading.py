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

import scipy.sparse

from scvae.data.loaders import LOADERS
from scvae.utilities import (
    format_duration, normalise_string,
    extension, download_file, copy_file
)


def acquire_data_set(title, urls, directory):

    paths = {}

    if not urls:
        return paths

    if not os.path.exists(directory):
        os.makedirs(directory)

    for values_or_labels in urls:
        paths[values_or_labels] = {}

        for kind in urls[values_or_labels]:

            url = urls[values_or_labels][kind]

            if not url:
                paths[values_or_labels][kind] = None
                continue

            url_filename = os.path.split(url)[-1]
            file_extension = extension(url_filename)

            filename = "-".join(
                map(normalise_string, [title, values_or_labels, kind]))
            path = os.path.join(directory, filename) + file_extension

            paths[values_or_labels][kind] = path

            if not os.path.isfile(path):

                if url.startswith("."):
                    raise Exception(
                        "Data set file have to be manually placed in "
                        "correct folder."
                    )
                if os.path.isfile(url):

                    print("Copying {} for {} set.".format(
                        values_or_labels, kind, title))
                    start_time = time()

                    copy_file(url, path)

                    duration = time() - start_time
                    print("Data set copied ({}).".format(
                        format_duration(duration)))
                    print()

                else:

                    print("Downloading {} for {} set.".format(
                        values_or_labels, kind, title))
                    start_time = time()

                    download_file(url, path)

                    duration = time() - start_time
                    print("Data set downloaded ({}).".format(
                        format_duration(duration)))
                    print()

    return paths


def load_original_data_set(paths, data_format):

    print("Loading original data set.")
    loading_time_start = time()

    if data_format is None:
        raise ValueError("Data format not specified.")
    elif data_format.startswith("tsv"):
        data_format = "matrix_ebf"

    load = LOADERS.get(data_format)

    if load is None:
        raise ValueError("Data format `{}` not recognised.".format(
            data_format))

    data_dictionary = load(paths=paths)

    loading_duration = time() - loading_time_start
    print("Original data set loaded ({}).".format(format_duration(
        loading_duration)))

    if not isinstance(data_dictionary["values"], scipy.sparse.csr_matrix):

        print()

        print("Converting data set value array to sparse matrix.")
        sparse_time_start = time()

        data_dictionary["values"] = scipy.sparse.csr_matrix(
            data_dictionary["values"])

        sparse_duration = time() - sparse_time_start
        print("Data set value array converted ({}).".format(format_duration(
            sparse_duration)))

    return data_dictionary
