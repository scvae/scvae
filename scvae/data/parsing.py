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

import json
import os

import importlib_resources as resources

from scvae.utilities import normalise_string, extension

DATA_FORMAT_INCLUDING_LABELS = ["loom"]


def parse_input(input_file_or_name):

    if input_file_or_name.endswith(".json"):

        json_path = input_file_or_name

        with open(json_path, "r") as json_file:
            data_set_dictionary = json.load(json_file)

        name = _base_name(json_path)

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
        filename = os.path.basename(file_path)
        file_extension = extension(filename)
        data_format = file_extension[1:] if file_extension else None
        name = _base_name(file_path)
        data_set_dictionary = {
            "values": file_path,
            "format": data_format
        }
    else:
        name = input_file_or_name
        name = normalise_string(name)
        data_set_dictionary = None

    return name, data_set_dictionary


def save_data_set_dictionary_as_json_file(
        data_set_dictionary, name, directory):

    json_path = os.path.join(directory, name + ".json")

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(json_path, "w") as json_file:
        json.dump(data_set_dictionary, json_file, indent="\t")


def find_data_set(name, directory):

    data_sets = _load_data_set_metadata()

    title = None
    data_set = None

    json_path = os.path.join(directory, name, name + ".json")
    if os.path.exists(json_path):
        title, data_set = _data_set_from_json_file(json_path)

    if not title:
        for data_set_title, data_set_specifications in data_sets.items():
            if normalise_string(data_set_title) == normalise_string(name):
                title = data_set_title
                data_set = data_set_specifications
                break

    if not title:
        raise KeyError("Data set not found.")

    return title, data_set


def _load_data_set_metadata():
    with resources.open_text("scvae.data", "data_sets.json") as metadata_file:
        data_sets = json.load(metadata_file)
    return data_sets


def _data_set_from_json_file(json_path):

    with open(json_path, "r") as json_file:
        data_set = json.load(json_file)

    title = data_set.get("title", _base_name(json_path))
    data_format = data_set.get("format")

    if "URLs" not in data_set:

        if "values" not in data_set:
            raise Exception(
                "JSON dictionary have to contain either a values entry with "
                "a URL or path to the file containing the value matrix or a "
                "URLs entry containing a dictionary of URLs to files "
                "containing values and optionally labels."
            )

        if data_format in DATA_FORMAT_INCLUDING_LABELS:
            urls = {
                "all": {
                    "full": data_set["values"]
                }
            }
        else:
            urls = {
                "values": {
                    "full": data_set["values"]
                }
            }

            if "labels" in data_set:
                urls["labels"] = {
                    "full": data_set["labels"]
                }

        data_set["URLs"] = urls

    return title, data_set


def _base_name(path):
    base_name = os.path.basename(path)
    base_name = base_name.split(os.extsep, 1)[0]
    return base_name
