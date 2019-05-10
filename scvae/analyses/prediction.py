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

from time import time

import numpy
import scipy.stats
from sklearn.cluster import KMeans, MiniBatchKMeans

from scvae.defaults import defaults
from scvae.utilities import normalise_string, proper_string, format_duration

PREDICTION_METHODS = {}

MAXIMUM_SAMPLE_SIZE_FOR_NORMAL_KMEANS = 10000


def predict_labels(training_set, evaluation_set, specifications=None,
                   method=None, number_of_clusters=None):

    if specifications is None:
        if method is None:
            method = defaults["evaluation"]["prediction_method"]
        specifications = PredictionSpecifications(
            method=method,
            number_of_clusters=number_of_clusters,
            training_set=training_set.kind
        )

    method = specifications.method
    number_of_clusters = specifications.number_of_clusters

    predict = PREDICTION_METHODS[specifications.method]["function"]

    print(
        "Predicting labels for evaluation set using {} with {} components."
        .format(method, number_of_clusters)
    )
    prediction_time_start = time()

    if evaluation_set.has_labels:

        class_names_to_class_ids = numpy.vectorize(
            lambda class_name:
            evaluation_set.class_name_to_class_id[class_name]
        )
        class_ids_to_class_names = numpy.vectorize(
            lambda class_id:
            evaluation_set.class_id_to_class_name[class_id]
        )

        evaluation_label_ids = class_names_to_class_ids(
            evaluation_set.labels)

        if evaluation_set.excluded_classes:
            excluded_class_ids = class_names_to_class_ids(
                evaluation_set.excluded_classes)
        else:
            excluded_class_ids = []

    if evaluation_set.has_superset_labels:

        superset_class_names_to_superset_class_ids = numpy.vectorize(
            lambda superset_class_name:
            evaluation_set.superset_class_name_to_superset_class_id[
                superset_class_name]
        )
        superset_class_ids_to_superset_class_names = numpy.vectorize(
            lambda superset_class_id:
            evaluation_set.superset_class_id_to_superset_class_name[
                superset_class_id]
        )

        evaluation_superset_label_ids = (
            superset_class_names_to_superset_class_ids(
                evaluation_set.superset_labels))

        if evaluation_set.excluded_superset_classes:
            excluded_superset_class_ids = (
                superset_class_names_to_superset_class_ids(
                    evaluation_set.excluded_superset_classes))
        else:
            excluded_superset_class_ids = []

    cluster_ids, predicted_labels, predicted_superset_labels = predict(
        training_set=training_set,
        evaluation_set=evaluation_set,
        number_of_clusters=number_of_clusters
    )

    if cluster_ids is not None:

        if predicted_labels is None and evaluation_set.has_labels:
            predicted_label_ids = map_cluster_ids_to_label_ids(
                evaluation_label_ids,
                cluster_ids,
                excluded_class_ids
            )
            predicted_labels = class_ids_to_class_names(predicted_label_ids)

        if (predicted_superset_labels is None
                and evaluation_set.has_superset_labels):
            predicted_superset_label_ids = map_cluster_ids_to_label_ids(
                evaluation_superset_label_ids,
                cluster_ids,
                excluded_superset_class_ids
            )
            predicted_superset_labels = (
                superset_class_ids_to_superset_class_names(
                    predicted_superset_label_ids))

    prediction_duration = time() - prediction_time_start
    print("Labels predicted ({}).".format(
        format_duration(prediction_duration)))

    return cluster_ids, predicted_labels, predicted_superset_labels


def map_cluster_ids_to_label_ids(label_ids, cluster_ids,
                                 excluded_class_ids=[]):
    unique_cluster_ids = numpy.unique(cluster_ids).tolist()
    predicted_label_ids = numpy.zeros_like(cluster_ids)
    for unique_cluster_id in unique_cluster_ids:
        indices = cluster_ids == unique_cluster_id
        index_labels = label_ids[indices]
        for excluded_class_id in excluded_class_ids:
            index_labels = index_labels[index_labels != excluded_class_id]
        if len(index_labels) == 0:
            continue
        predicted_label_ids[indices] = scipy.stats.mode(index_labels)[0]
    return predicted_label_ids


class PredictionSpecifications():
    def __init__(self, method, number_of_clusters=None,
                 training_set_kind=None):

        prediction_method_names = {
                name: specifications["aliases"]
                for name, specifications in PREDICTION_METHODS.items()
            }
        method = proper_string(method, prediction_method_names)

        if method not in PREDICTION_METHODS:
            raise ValueError(
                "Prediction method `{}` not found.".format(method))

        if number_of_clusters is None:
            raise TypeError("Number of clusters not set.")

        self.method = method
        self.number_of_clusters = number_of_clusters

        if training_set_kind:
            training_set_kind = normalise_string(training_set_kind)
        self.training_set_kind = training_set_kind

    @property
    def name(self):
        name_parts = [
            self.method, self.number_of_clusters]
        if self.training_set_kind and self.training_set_kind != "training":
            name_parts.append(self.training_set_kind)
        name = "_".join(map(
            lambda s: normalise_string(str(s)).replace("_", ""),
            name_parts
        ))
        return name


def _register_prediction_method(name):
    def decorator(function):
        aliases = set()
        alias = normalise_string(name)
        aliases.add(alias)
        alias = alias.replace("_", "")
        aliases.add(alias)
        PREDICTION_METHODS[name] = {
            "aliases": aliases,
            "function": function
        }
        return function
    return decorator


@_register_prediction_method("k-means")
def _predict_using_kmeans(training_set, evaluation_set, number_of_clusters):

    if (training_set.number_of_examples
            <= MAXIMUM_SAMPLE_SIZE_FOR_NORMAL_KMEANS):
        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=None
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=number_of_clusters,
            random_state=None,
            batch_size=100
        )

    model.fit(training_set.values)
    cluster_ids = model.predict(evaluation_set.values)

    predicted_labels = None
    predicted_superset_labels = None

    return cluster_ids, predicted_labels, predicted_superset_labels


@_register_prediction_method("model")
def _predict_using_model(training_set, evaluation_set, number_of_clusters):

    cluster_ids = evaluation_set.cluster_ids
    predicted_labels = evaluation_set.predicted_labels
    predicted_superset_labels = evaluation_set.predicted_superset_labels

    return cluster_ids, predicted_labels, predicted_superset_labels
