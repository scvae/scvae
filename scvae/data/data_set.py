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
import re
import shutil
from time import time

import numpy
import seaborn

from scvae.data import internal_io, loading, parsing, processing, sparse
from scvae.defaults import defaults
from scvae.utilities import format_duration, normalise_string

PREPROCESS_SUFFIX = "preprocessed"
ORIGINAL_SUFFIX = "original"
PREPROCESSED_EXTENSION = ".sparse.h5"

MINIMUM_NUMBER_OF_SECONDS_BEFORE_SAVING = 30

DEFAULT_TERMS = {
    "example": "example",
    "feature": "feature",
    "mapped feature": "mapped feature",
    "class": "class",
    "type": "value",
    "item": "item"
}
DEFAULT_EXCLUDED_CLASSES = ["No class"]

GENERIC_CLASS_NAMES = ["Others", "Unknown", "No class", "Remaining"]


class DataSet:
    """Data set class for working with scVAE.

    To easily handle values, labels, metadata, and so on for data sets,
    scVAE uses this class. Other data formats will have to be converted
    to it.

    Arguments:
        input_file_or_name (str): Path to a data set file or a title for
            a supported data set (see :ref:`Data sets`).
        data_format (str, optional): Format used to store data set (see
            :ref:`Custom data sets`).
        title (str, optional): Title of data set for use in, e.g.,
            plots.
        specifications (dict, optional): Metadata for data set.
        values (2-d NumPy array, optional): Matrix for (count) values
            with rows representing examples/cells and columns
            features/genes.
        labels (1-d NumPy array, optional): List of labels for
            examples/cells in the same order as for ``values``.
        example_names (1-d NumPy array, optional): List of names for
            examples/cells in the same order as for ``values``.
        feature_names (1-d NumPy array, optional): List of names for
            features/genes in the same order as for ``values``.
        batch_indices (1-d NumPy array, optional): List of batch
            indices for examples/cells in the same order as for
            ``values``.
        feature_selection (list, optional): Method and parameters for
            feature selection in a list.
        example_filter (list, optional): Method and parameters for
            example filtering in a list.
        preprocessing_methods (list, optional): Ordered list of
            preprocessing methods applied to (count) values:
            ``"normalise"`` (each feature/gene), ``"log"``, and
            ``"exp"``.
        directory (str, optional): Directory where data set is saved.

    Attributes:
        name: Short name for data set used in filenames.
        title: Title of data set for use in, e.g., plots.
        specifications: Metadata for data set. If a JSON file was
            provided, this would contain the contents.
        data_format: Format used to store data set.
        terms: Dictionary of terms to use for, e.g., ``"example"``
            (cell), ``"feature"`` (gene), and ``"class"`` (cell type).
        values: Matrix for (count) values with rows representing
            examples/cells and columns features/genes.
        labels: List of labels for examples/cells in the same order as
            for `values`.
        example_names: List of names for examples/cells in the same
            order as for `values`.
        feature_names: List of names for features/genes in the same
            order as for `values`.
        batch_indices: List of batch indices for examples/cells in the
            same order as for `values`.
        number_of_examples: The number of examples/cells.
        number_of_features: The number of features/genes.
        number_of_classes: The number of classes/cell types.
        feature_selection_method: The method used for selecting
            features.
        feature_selection_parameters: List of parameters for the
            feature selection method.
        example_filter_method: The method used for filtering examples.
        example_filter_parameters: List of parameters for the example
            filtering method.
        kind: The kind of data set: ``"full"``, ``"training"``,
            ``"validation"``, or ``"test"``.
        version: The version of the data set: ``"original"``,
            ``"reconstructed"``, or latent (``"z"`` or ``"y"``).
    """

    def __init__(self,
                 input_file_or_name,
                 data_format=None,
                 title=None,
                 specifications=None,
                 values=None,
                 labels=None,
                 example_names=None,
                 feature_names=None,
                 batch_indices=None,
                 feature_selection=None,
                 example_filter=None,
                 preprocessing_methods=None,
                 directory=None,
                 **kwargs):

        super().__init__()

        # Name of data set and optional entry for data sets dictionary
        self.name, data_set_dictionary = parsing.parse_input(
            input_file_or_name)

        # Directories and paths for data set
        if directory is None:
            directory = defaults["data"]["directory"]
        self._directory = os.path.join(directory, self.name)
        self._preprocess_directory = os.path.join(
            self._directory, PREPROCESS_SUFFIX)
        self._original_directory = os.path.join(
            self._directory, ORIGINAL_SUFFIX)

        # Save data set dictionary if necessary
        if data_set_dictionary:
            if os.path.exists(self._directory):
                shutil.rmtree(self._directory)
            parsing.save_data_set_dictionary_as_json_file(
                data_set_dictionary,
                self.name,
                self._directory
            )

        # Find data set
        if title is None or specifications is None:
            parsed_title, parsed_specifications = parsing.find_data_set(
                self.name, directory)
        if title is None:
            title = parsed_title
        if specifications is None:
            specifications = parsed_specifications
        self.title = title
        self.specifications = specifications

        # Prioritise data format from metadata
        data_format_from_metadata = self.specifications.get("format")
        if data_format is None:
            data_format = defaults["data"]["format"]
        if data_format == "infer":
            data_format = data_format_from_metadata
        else:
            data_format = normalise_string(data_format)
            if (data_format_from_metadata
                    and data_format_from_metadata != data_format):
                raise ValueError(
                    "Data format already specified in metadata "
                    "and cannot be changed (is `{}`; wanted `{}`).".format(
                        data_format_from_metadata,
                        data_format
                    )
                )
        self.data_format = data_format

        # Terms for data set
        self.terms = _postprocess_terms(self.specifications.get(
            "terms", DEFAULT_TERMS))

        # Example type for data set
        self.example_type = self.specifications.get("example type", "unknown")

        # Discreteness
        self.discreteness = self.example_type == "counts"

        # Feature dimensions for data set
        self.feature_dimensions = self.specifications.get(
            "feature dimensions")

        # Label super set for data set
        self.label_superset = self.specifications.get("label superset")
        self.superset_labels = None
        self.number_of_superset_classes = None

        # Label palette for data set
        self.class_palette = self.specifications.get("class palette")
        self.superset_class_palette = _create_superset_class_palette(
            self.class_palette, self.label_superset)

        # Excluded classes for data set
        self.excluded_classes = self.specifications.get("excluded classes", [])

        # Excluded classes for data set
        self.excluded_superset_classes = self.specifications.get(
            "excluded superset classes", [])

        # Unpack keyword arguments
        total_standard_deviations = kwargs.get("total_standard_deviations")
        explained_standard_deviations = kwargs.get(
            "explained_standard_deviations")
        preprocessed_values = kwargs.get("preprocessed_values")
        binarised_values = kwargs.get("binarised_values")
        class_names = kwargs.get("class_names")
        batch_names = kwargs.get("batch_names")
        features_mapped = kwargs.get("features_mapped", False)
        preprocessed = kwargs.get("preprocessed")
        noisy_preprocessing_methods = kwargs.get("noisy_preprocessing_methods")
        binarise_values = kwargs.get("binarise_values", False)
        kind = kwargs.get("kind", "full",)
        version = kwargs.get("version", "original")

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
        self.batch_indices = None
        self.batch_names = None
        self.number_of_batches = None
        self.class_names = None
        self.number_of_examples = None
        self.number_of_features = None
        self.number_of_classes = None
        self.update(
            values=values,
            total_standard_deviations=total_standard_deviations,
            explained_standard_deviations=explained_standard_deviations,
            preprocessed_values=preprocessed_values,
            binarised_values=binarised_values,
            labels=labels,
            class_names=class_names,
            example_names=example_names,
            feature_names=feature_names,
            batch_indices=batch_indices,
            batch_names=batch_names
        )

        # Predicted labels
        self.prediction_specifications = None
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
        self.label_sorter = _create_label_sorter(sorted_class_names)
        sorted_superset_class_names = self.specifications.get(
            "sorted superset class names")
        self.superset_label_sorter = _create_label_sorter(
            sorted_superset_class_names)

        # Feature mapping
        map_features = kwargs.get("map_features")
        if map_features is None:
            map_features = defaults["data"]["map_features"]
        self.map_features = map_features
        self.feature_mapping = None
        self.features_mapped = features_mapped

        if self.features_mapped:
            self.terms = _update_tag_for_mapped_features(self.terms)

        # Feature selection
        if feature_selection is None:
            feature_selection = defaults["data"]["feature_selection"]
        self.feature_selection = feature_selection
        if self.feature_selection:
            self.feature_selection_method = self.feature_selection[0]
            if len(self.feature_selection) > 1:
                self.feature_selection_parameters = self.feature_selection[1:]
            else:
                self.feature_selection_parameters = None
        else:
            self.feature_selection_method = None
            self.feature_selection_parameters = None

        # Example filterering
        if example_filter is None:
            example_filter = defaults["data"]["example_filter"]
        self.example_filter = example_filter
        if self.example_filter:
            self.example_filter_method = self.example_filter[0]
            if len(self.example_filter) > 1:
                self.example_filter_parameters = self.example_filter[1:]
            else:
                self.example_filter_parameters = None
        else:
            self.example_filter_method = None
            self.example_filter_parameters = None

        # Preprocessing methods
        if preprocessing_methods is None:
            preprocessing_methods = defaults["data"]["preprocessing_methods"]
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

        # Noisy preprocessing
        if noisy_preprocessing_methods is None:
            noisy_preprocessing_methods = defaults["data"][
                "noisy_preprocessing_methods"]
        if self.preprocessed:
            noisy_preprocessing_methods = []
        self.noisy_preprocessing_methods = noisy_preprocessing_methods

        if self.noisy_preprocessing_methods:
            self.noisy_preprocess = processing.build_preprocessor(
                self.noisy_preprocessing_methods,
                noisy=True
            )
        else:
            self.noisy_preprocess = None

        if self.kind == "full" and self.values is None:

            print("Data set:")
            print("    title:", self.title)

            if self.map_features:
                print("    feature mapping: if available")

            if self.feature_selection_method:
                print("    feature selection:", self.feature_selection_method)
                if self.feature_selection_parameters:
                    print(
                        "        parameters:",
                        ", ".join(self.feature_selection_parameters)
                    )
                else:
                    print("        parameters: default")
            else:
                print("    feature selection: none")

            if self.example_filter_method:
                print("    example filter:", self.example_filter_method)
                if self.example_filter_parameters:
                    print(
                        "        parameter(s):",
                        ", ".join(self.example_filter_parameters)
                    )
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
        """Total number of (count) values in matrix."""
        return self.number_of_examples * self.number_of_features

    @property
    def class_probabilities(self):

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
    def has_batches(self):
        return self.batch_indices is not None

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

    @property
    def default_feature_parameters(self):

        feature_selection_parameters = None

        if self.feature_selection_method:
            feature_selection = normalise_string(self.feature_selection_method)

            if feature_selection == "keep_variances_above":
                feature_selection_parameters = [0.5]

            elif feature_selection == "keep_highest_variances":
                if self.number_of_features is not None:
                    feature_selection_parameters = [
                        int(self.number_of_features / 2)
                    ]

        return feature_selection_parameters

    @property
    def default_splitting_method(self):
        if self.split_indices:
            return "indices"
        else:
            return "random"

    def update(self, values=None,
               total_standard_deviations=None,
               explained_standard_deviations=None,
               preprocessed_values=None, binarised_values=None,
               labels=None, class_names=None,
               example_names=None, feature_names=None,
               batch_indices=None, batch_names=None):

        if values is not None:

            self.values = values

            self.count_sum = self.values.sum(axis=1).reshape(-1, 1)
            if isinstance(self.count_sum, numpy.matrix):
                self.count_sum = self.count_sum.A
            self.normalised_count_sum = self.count_sum / self.count_sum.max()

            n_examples_from_values, n_featues_from_values = values.shape

            if example_names is not None:
                self.example_names = example_names
                if self.example_names.ndim > 1:
                    raise ValueError(
                        "The list of example names is multi-dimensional: {}."
                        .format(self.example_names.shape)
                    )
                n_examples = self.example_names.shape[0]
                if n_examples_from_values != n_examples:
                    raise ValueError(
                        "The number of examples ({}) in the value matrix "
                        "is not the same as the number of example names ({})."
                        .format(n_examples_from_values, n_examples)
                    )

            if feature_names is not None:
                self.feature_names = feature_names
                if self.feature_names.ndim > 1:
                    raise ValueError(
                        "The list of feature names is multi-dimensional: {}."
                        .format(self.feature_names.shape)
                    )
                n_features = self.feature_names.shape[0]
                if n_featues_from_values != n_features:
                    raise ValueError(
                        "The number of features in the value matrix ({}) "
                        "is not the same as the number of feature names ({})."
                        .format(n_featues_from_values, n_features)
                    )

            self.number_of_examples = n_examples_from_values
            self.number_of_features = n_featues_from_values

        else:

            if example_names is not None and feature_names is not None:

                self.example_names = example_names
                self.feature_names = feature_names

        if labels is not None:

            if issubclass(labels.dtype.type, numpy.float):
                labels_int = labels.astype(int)
                if (labels == labels_int).all():
                    labels = labels_int

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
                for excluded_class in DEFAULT_EXCLUDED_CLASSES:
                    if excluded_class in self.class_names:
                        self.excluded_classes.append(excluded_class)

            self.number_of_classes = len(self.class_names)

            if self.excluded_classes:
                self.number_of_excluded_classes = len(self.excluded_classes)
            else:
                self.number_of_excluded_classes = 0

            if self.class_palette is None:
                self.class_palette = _create_class_palette(self.class_names)

            if self.label_superset:

                self.superset_labels = _map_labels_to_superset_labels(
                    self.labels, self.label_superset)

                self.superset_class_names = numpy.unique(
                    self.superset_labels).tolist()

                self.superset_class_id_to_superset_class_name = {}
                self.superset_class_name_to_superset_class_id = {}

                for i, class_name in enumerate(self.superset_class_names):
                    self.superset_class_name_to_superset_class_id[
                        class_name] = i
                    self.superset_class_id_to_superset_class_name[
                        i] = class_name

                if not self.excluded_superset_classes:
                    for excluded_class in DEFAULT_EXCLUDED_CLASSES:
                        if excluded_class in self.superset_class_names:
                            self.excluded_superset_classes.append(
                                excluded_class)

                self.number_of_superset_classes = len(
                    self.superset_class_names)

                if self.excluded_superset_classes:
                    self.number_of_excluded_superset_classes = len(
                        self.excluded_superset_classes)
                else:
                    self.number_of_excluded_superset_classes = 0

                if self.superset_class_palette is None:
                    self.superset_class_palette = (
                        _create_superset_class_palette(
                            self.class_names, self.label_superset
                        )
                    )

        if total_standard_deviations is not None:
            self.total_standard_deviations = total_standard_deviations

        if explained_standard_deviations is not None:
            self.explained_standard_deviations = explained_standard_deviations

        if preprocessed_values is not None:
            self.preprocessed_values = preprocessed_values

        if binarised_values is not None:
            self.binarised_values = binarised_values

        if batch_indices is not None:
            batch_indices_int = batch_indices.astype(int)
            if (batch_indices == batch_indices_int).all():
                batch_indices = batch_indices_int
            else:
                raise TypeError("Batch indices should be integers.")
            self.batch_indices = batch_indices.reshape(-1, 1)

            if batch_names is None:
                batch_names = numpy.unique(self.batch_indices)
            self.batch_names = batch_names

            self.number_of_batches = len(self.batch_names)

    def update_predictions(self,
                           prediction_specifications=None,
                           predicted_cluster_ids=None,
                           predicted_labels=None,
                           predicted_class_names=None,
                           predicted_superset_labels=None,
                           predicted_superset_class_names=None):

        if prediction_specifications is not None:
            self.prediction_specifications = prediction_specifications

        if predicted_cluster_ids is not None:
            self.predicted_cluster_ids = predicted_cluster_ids

        if predicted_labels is not None:

            self.predicted_labels = predicted_labels

            if predicted_class_names is not None:
                self.predicted_class_names = predicted_class_names
            else:
                self.predicted_class_names = numpy.unique(
                    self.predicted_labels).tolist()

            self.number_of_predicted_classes = len(self.predicted_class_names)

            if set(self.predicted_class_names) < set(self.class_names):
                self.predicted_class_palette = self.class_palette
                self.predicted_label_sorter = self.label_sorter

        if predicted_superset_labels is not None:

            self.predicted_superset_labels = predicted_superset_labels

            if predicted_superset_class_names is not None:
                self.predicted_superset_class_names = (
                    predicted_superset_class_names)
            else:
                self.predicted_superset_class_names = numpy.unique(
                    self.predicted_superset_labels).tolist()

            self.number_of_predicted_superset_classes = len(
                self.predicted_superset_class_names)

            if set(self.predicted_superset_class_names) < set(
                    self.superset_class_names):
                self.predicted_superset_class_palette = (
                    self.superset_class_palette)
                self.predicted_superset_label_sorter = (
                    self.superset_label_sorter)

    def reset_predictions(self):

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
        """Load data set."""

        sparse_path = self._build_preprocessed_path()

        if os.path.isfile(sparse_path):
            print("Loading data set.")
            data_dictionary = internal_io.load_data_dictionary(
                path=sparse_path)
            print()
        else:
            urls = self.specifications.get("URLs", None)
            original_paths = loading.acquire_data_set(
                title=self.title,
                urls=urls,
                directory=self._original_directory
            )

            loading_time_start = time()
            data_dictionary = loading.load_original_data_set(
                paths=original_paths,
                data_format=self.data_format
            )
            loading_duration = time() - loading_time_start

            print()

            if loading_duration > MINIMUM_NUMBER_OF_SECONDS_BEFORE_SAVING:
                if not os.path.exists(self._preprocess_directory):
                    os.makedirs(self._preprocess_directory)

                print("Saving data set.")
                internal_io.save_data_dictionary(
                    data_dictionary=data_dictionary,
                    path=sparse_path
                )

                print()

        data_dictionary["values"] = sparse.SparseRowMatrix(
            data_dictionary["values"])

        self.update(
            values=data_dictionary["values"],
            labels=data_dictionary["labels"],
            example_names=data_dictionary["example names"],
            feature_names=data_dictionary["feature names"],
            batch_indices=data_dictionary.get("batch indices")
        )

        self.split_indices = data_dictionary.get("split indices")
        self.feature_mapping = data_dictionary.get("feature mapping")

        if self.feature_mapping is None:
            self.map_features = False

        if not self.feature_selection_parameters:
            self.feature_selection_parameters = self.default_feature_parameters

        self.preprocess()

        if self.binarise_values:
            self.binarise()

    def preprocess(self):

        if (not self.map_features and not self.preprocessing_methods
                and not self.feature_selection and not self.example_filter):
            self.update(preprocessed_values=None)
            return

        sparse_path = self._build_preprocessed_path(
            map_features=self.map_features,
            preprocessing_methods=self.preprocessing_methods,
            feature_selection_method=self.feature_selection_method,
            feature_selection_parameters=self.feature_selection_parameters,
            example_filter_method=self.example_filter_method,
            example_filter_parameters=self.example_filter_parameters
        )

        if os.path.isfile(sparse_path):
            print("Loading preprocessed data.")
            data_dictionary = internal_io.load_data_dictionary(sparse_path)
            if "preprocessed values" not in data_dictionary:
                data_dictionary["preprocessed values"] = None
            if self.map_features:
                self.features_mapped = True
                self.terms = _update_tag_for_mapped_features(self.terms)
            print()
        else:

            preprocessing_time_start = time()

            values = self.values
            example_names = self.example_names
            feature_names = self.feature_names

            if self.map_features and not self.features_mapped:

                print(
                    "Mapping {} original features to {} new features."
                    .format(self.number_of_features, len(self.feature_mapping))
                )
                start_time = time()

                values, feature_names = processing.map_features(
                    values, feature_names, self.feature_mapping)

                self.features_mapped = True
                self.terms = _update_tag_for_mapped_features(self.terms)

                duration = time() - start_time
                print("Features mapped ({}).".format(format_duration(
                    duration)))

                print()

            if not self.preprocessed and self.preprocessing_methods:

                print("Preprocessing values.")
                start_time = time()

                preprocessing_function = processing.build_preprocessor(
                    self.preprocessing_methods)
                preprocessed_values = preprocessing_function(values)

                duration = time() - start_time
                print(
                    "Values preprocessed ({})."
                    .format(format_duration(duration))
                )

                print()

            else:
                preprocessed_values = None

            if self.feature_selection:
                values_dictionary, feature_names = processing.select_features(
                    {"original": values,
                     "preprocessed": preprocessed_values},
                    self.feature_names,
                    method=self.feature_selection_method,
                    parameters=self.feature_selection_parameters
                )

                values = values_dictionary["original"]
                preprocessed_values = values_dictionary["preprocessed"]

                print()

            if self.example_filter:
                values_dictionary, example_names, labels, batch_indices = (
                    processing.filter_examples(
                        {"original": values,
                         "preprocessed": preprocessed_values},
                        self.example_names,
                        method=self.example_filter_method,
                        parameters=self.example_filter_parameters,
                        labels=self.labels,
                        excluded_classes=self.excluded_classes,
                        superset_labels=self.superset_labels,
                        excluded_superset_classes=(
                            self.excluded_superset_classes),
                        batch_indices=self.batch_indices,
                        count_sum=self.count_sum
                    )
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
                data_dictionary["batch indices"] = batch_indices

            preprocessing_duration = time() - preprocessing_time_start

            if (preprocessing_duration
                    > MINIMUM_NUMBER_OF_SECONDS_BEFORE_SAVING):

                if not os.path.exists(self._preprocess_directory):
                    os.makedirs(self._preprocess_directory)

                print("Saving preprocessed data set.")
                internal_io.save_data_dictionary(data_dictionary, sparse_path)
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
            batch_indices = data_dictionary["batch indices"]
        else:
            example_names = self.example_names
            labels = self.labels
            batch_indices = self.batch_indices

        values = sparse.SparseRowMatrix(values)
        preprocessed_values = sparse.SparseRowMatrix(preprocessed_values)

        self.update(
            values=values,
            preprocessed_values=preprocessed_values,
            example_names=example_names,
            feature_names=feature_names,
            labels=labels,
            batch_indices=batch_indices
        )

    def binarise(self):

        if self.preprocessed_values is None:
            raise NotImplementedError(
                "Data set values have to have been preprocessed and feature"
                " selected first."
            )

        binarise_preprocessing = ["binarise"]

        sparse_path = self._build_preprocessed_path(
            map_features=self.map_features,
            preprocessing_methods=binarise_preprocessing,
            feature_selection_method=self.feature_selection_method,
            feature_selection_parameters=self.feature_selection_parameters,
            example_filter_method=self.example_filter_method,
            example_filter_parameters=self.example_filter_parameters
        )

        if os.path.isfile(sparse_path):
            print("Loading binarised data.")
            data_dictionary = internal_io.load_data_dictionary(sparse_path)

        else:

            binarising_time_start = time()

            if self.preprocessing_methods != binarise_preprocessing:

                print("Binarising values.")
                start_time = time()

                binarisation_function = processing.build_preprocessor(
                    binarise_preprocessing)
                binarised_values = binarisation_function(self.values)

                duration = time() - start_time
                print(
                    "Values binarised ({}).".format(format_duration(duration))
                )

                print()

            elif self.preprocessing_methods == binarise_preprocessing:
                binarised_values = self.preprocessed_values

            data_dictionary = {
                "values": self.values,
                "preprocessed values": binarised_values,
                "feature names": self.feature_names
            }

            binarising_duration = time() - binarising_time_start

            if binarising_duration > MINIMUM_NUMBER_OF_SECONDS_BEFORE_SAVING:

                if not os.path.exists(self._preprocess_directory):
                    os.makedirs(self._preprocess_directory)

                print("Saving binarised data set.")
                internal_io.save_data_dictionary(data_dictionary, sparse_path)

        binarised_values = sparse.SparseRowMatrix(binarised_values)

        self.update(binarised_values=data_dictionary["preprocessed values"])

    def split(self, method=None, fraction=None):
        """Split data set into subsets.

        The data set is split into a training set to train a model, a
        validation set to validate the model during training, and a
        test set to evaluate the model after training.

        Arguments:
            method (str, optional): The method to use: ``"random"`` or
                ``"sequential"``.
            fraction (float, optional): The fraction to use for
                training and, optionally, validation.

        Returns:
            Training, validation, and test sets.

        """

        if method is None:
            method = defaults["data"]["splitting_method"]
        if fraction is None:
            fraction = defaults["data"]["splitting_fraction"]

        if method == "default":
            method = self.default_splitting_method

        sparse_path = self._build_preprocessed_path(
            map_features=self.map_features,
            preprocessing_methods=self.preprocessing_methods,
            feature_selection_method=self.feature_selection_method,
            feature_selection_parameters=self.feature_selection_parameters,
            example_filter_method=self.example_filter_method,
            example_filter_parameters=self.example_filter_parameters,
            splitting_method=method,
            splitting_fraction=fraction,
            split_indices=self.split_indices
        )

        print("Splitting:")
        print("    method:", method)
        if method != "indices":
            print("    fraction: {:.1f} %".format(100 * fraction))
        print()

        if os.path.isfile(sparse_path):
            print("Loading split data sets.")
            split_data_dictionary = internal_io.load_data_dictionary(
                path=sparse_path)
            if self.map_features:
                self.features_mapped = True
                self.terms = _update_tag_for_mapped_features(self.terms)
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
                "batch indices": self.batch_indices,
                "class names": self.class_names,
                "split indices": self.split_indices
            }

            splitting_time_start = time()
            split_data_dictionary = processing.split_data_set(
                data_dictionary, method, fraction)
            splitting_duration = time() - splitting_time_start

            print()

            if splitting_duration > MINIMUM_NUMBER_OF_SECONDS_BEFORE_SAVING:

                if not os.path.exists(self._preprocess_directory):
                    os.makedirs(self._preprocess_directory)

                print("Saving split data sets.")
                internal_io.save_data_dictionary(
                    data_dictionary=split_data_dictionary,
                    parse=sparse_path
                )
                print()

        for data_subset in split_data_dictionary:
            if not isinstance(split_data_dictionary[data_subset], dict):
                continue
            for data_subset_key in split_data_dictionary[data_subset]:
                if "values" in data_subset_key:
                    values = split_data_dictionary[data_subset][
                        data_subset_key]
                    if values is not None:
                        split_data_dictionary[data_subset][data_subset_key] = (
                            sparse.SparseRowMatrix(values))

        training_set = DataSet(
            self.name,
            title=self.title,
            specifications=self.specifications,
            values=split_data_dictionary["training set"]["values"],
            preprocessed_values=(
                split_data_dictionary["training set"]["preprocessed values"]),
            binarised_values=(
                split_data_dictionary["training set"]["binarised values"]),
            labels=split_data_dictionary["training set"]["labels"],
            example_names=(
                split_data_dictionary["training set"]["example names"]),
            feature_names=split_data_dictionary["feature names"],
            batch_indices=(
                split_data_dictionary["training set"]["batch indices"]),
            batch_names=self.batch_names,
            features_mapped=self.features_mapped,
            class_names=split_data_dictionary["class names"],
            feature_selection=self.feature_selection,
            example_filter=self.example_filter,
            preprocessing_methods=self.preprocessing_methods,
            noisy_preprocessing_methods=self.noisy_preprocessing_methods,
            kind="training"
        )

        validation_set = DataSet(
            self.name,
            title=self.title,
            specifications=self.specifications,
            values=split_data_dictionary["validation set"]["values"],
            preprocessed_values=(
                split_data_dictionary["validation set"]["preprocessed values"]
            ),
            binarised_values=(
                split_data_dictionary["validation set"]["binarised values"]),
            labels=split_data_dictionary["validation set"]["labels"],
            example_names=(
                split_data_dictionary["validation set"]["example names"]),
            feature_names=split_data_dictionary["feature names"],
            batch_indices=(
                split_data_dictionary["training set"]["batch indices"]),
            batch_names=self.batch_names,
            features_mapped=self.features_mapped,
            class_names=split_data_dictionary["class names"],
            feature_selection=self.feature_selection,
            example_filter=self.example_filter,
            preprocessing_methods=self.preprocessing_methods,
            noisy_preprocessing_methods=self.noisy_preprocessing_methods,
            kind="validation"
        )

        test_set = DataSet(
            self.name,
            title=self.title,
            specifications=self.specifications,
            values=split_data_dictionary["test set"]["values"],
            preprocessed_values=(
                split_data_dictionary["test set"]["preprocessed values"]),
            binarised_values=(
                split_data_dictionary["test set"]["binarised values"]),
            labels=split_data_dictionary["test set"]["labels"],
            example_names=split_data_dictionary["test set"]["example names"],
            feature_names=split_data_dictionary["feature names"],
            batch_indices=(
                split_data_dictionary["training set"]["batch indices"]),
            batch_names=self.batch_names,
            features_mapped=self.features_mapped,
            class_names=split_data_dictionary["class names"],
            feature_selection=self.feature_selection,
            example_filter=self.example_filter,
            preprocessing_methods=self.preprocessing_methods,
            noisy_preprocessing_methods=self.noisy_preprocessing_methods,
            kind="test"
        )

        print(
            "Data sets with {} features{}{}:\n".format(
                training_set.number_of_features,
                (" and {} classes".format(self.number_of_classes)
                    if self.number_of_classes else ""),
                (" ({} superset classes)".format(
                    self.number_of_superset_classes)
                    if self.number_of_superset_classes else "")
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

    def clear(self):
        """Clear data set."""

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
        self.batch_indices = None
        self.batch_names = None
        self.number_of_batches = None
        self.class_names = None
        self.number_of_examples = None
        self.number_of_features = None
        self.number_of_classes = None

    def _build_preprocessed_path(
            self,
            map_features=None,
            preprocessing_methods=None,
            feature_selection_method=None,
            feature_selection_parameters=None,
            example_filter_method=None,
            example_filter_parameters=None,
            splitting_method=None,
            splitting_fraction=None,
            split_indices=None):

        base_path = os.path.join(self._preprocess_directory, self.name)

        filename_parts = [base_path]

        if map_features:
            filename_parts.append("features_mapped")

        if feature_selection_method:
            feature_selection_part = normalise_string(feature_selection_method)
            if feature_selection_parameters:
                for parameter in feature_selection_parameters:
                    feature_selection_part += "_" + normalise_string(str(
                        parameter))
            filename_parts.append(feature_selection_part)

        if example_filter_method:
            example_filter_part = normalise_string(example_filter_method)
            if example_filter_parameters:
                for parameter in example_filter_parameters:
                    example_filter_part += "_" + normalise_string(str(
                        parameter))
            filename_parts.append(example_filter_part)

        if preprocessing_methods:
            filename_parts.extend(map(normalise_string, preprocessing_methods))

        if splitting_method:
            filename_parts.append("split")

            if (splitting_method == "indices" and
                    len(split_indices) == 3 or not splitting_fraction):
                filename_parts.append(splitting_method)
            else:
                filename_parts.append("{}_{}".format(
                    splitting_method,
                    splitting_fraction
                ))

        path = "-".join(filename_parts) + PREPROCESSED_EXTENSION

        return path


def _postprocess_terms(terms):
    if "item" in terms and terms["item"]:
        value_tag = terms["item"] + " " + terms["type"]
    else:
        value_tag = terms["type"]
    terms["value"] = value_tag
    return terms


def _update_tag_for_mapped_features(terms):
    mapped_feature_tag = terms.pop("mapped feature", None)
    if mapped_feature_tag:
        terms["feature"] = mapped_feature_tag
    return terms


def _map_labels_to_superset_labels(labels, label_superset):

    if not label_superset:
        superset_labels = None

    elif label_superset == "infer":
        superset_labels = []

        for label in labels:
            superset_label = re.match("^( ?[A-Za-z])+", label).group()
            superset_labels.append(superset_label)

        superset_labels = numpy.array(superset_labels)

    else:
        label_superset_reverse = {
            v: k for k, vs in label_superset.items() for v in vs
        }
        labels_to_superset_labels = numpy.vectorize(
            lambda label: label_superset_reverse[label]
        )
        superset_labels = labels_to_superset_labels(labels)

    return superset_labels


def _create_class_palette(class_names):

    brewer_palette = seaborn.color_palette("Set3")

    if len(class_names) <= len(brewer_palette):
        class_palette = {
            c: brewer_palette[i] for i, c in enumerate(class_names)
        }
    else:
        class_palette = None

    return class_palette


def _create_superset_class_palette(class_palette, label_superset):

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
            superset_class_palette[superset_label] = numpy.array(
                superset_label_colours).mean(axis=0).tolist()

    return superset_class_palette


def _create_label_sorter(sorted_class_names=None):

    if not sorted_class_names:
        sorted_class_names = []

    def sort_key_for_label(label):

        label = str(label)

        if label.isdigit():
            number = int(label)
        elif label.isdecimal():
            number = float(label)
        else:
            number = numpy.nan

        n_sorted = len(sorted_class_names)
        n_generic = len(GENERIC_CLASS_NAMES)

        if label in sorted_class_names:
            index = sorted_class_names.index(label)
        elif label in GENERIC_CLASS_NAMES:
            index = n_sorted + GENERIC_CLASS_NAMES.index(label)
        else:
            index = n_sorted + n_generic

        sort_key = [number, index, label]

        return sort_key

    return sort_key_for_label
