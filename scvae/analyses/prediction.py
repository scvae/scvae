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
# TODO Remove DBSCAN and knee method
from kneed import KneeLocator
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

from scvae.utilities import proper_string, format_duration

PREDICTION_METHOD_NAMES = {
    "k-means": ["k_means", "kmeans"],
    "DBSCAN": ["dbscan"],
    "model": ["model"],
    "test": ["test"],
    "copy": ["copy"]
}

PREDICTION_METHOD_SPECIFICATIONS = {
    "k-means": {
        "inference": "inductive",
        "base": "centroids",
        "fixed number of clusters": True,
        "cluster kind": "centroid"
    },
    "DBSCAN": {
        "inference": "transductive",
        "base": "density",
        "fixed number of clusters": False
    },
    "model": {
        "inference": "inductive",
        "base": "distribution",
        "fixed number of clusters": True,
        "cluster kind": "component"
    }
}

MAXIMUM_SAMPLE_SIZE_FOR_NORMAL_KMEANS = 10000


def predict(training_set, evaluation_set, method="copy",
            number_of_clusters=2):

    method = proper_string(method, PREDICTION_METHOD_NAMES)

    specifications = PREDICTION_METHOD_SPECIFICATIONS.get(method, {})
    inference = specifications.get("inference", None)
    fixed_number_of_clusters = specifications.get(
        "fixed number of clusters", None)
    cluster_kind = specifications.get("cluster kind", None)

    prediction_string_parts = [
        "Predicting labels for evaluation set using {}".format(method)
    ]

    if fixed_number_of_clusters is not None and fixed_number_of_clusters:
        prediction_string_parts.append(
            "with {} {}s".format(number_of_clusters, cluster_kind)
        )

    if inference and inference == "inductive":
        prediction_string_parts.append(
            "on {} values for prediction training set"
            .format(training_set.version)
        )

    prediction_string = " ".join(prediction_string_parts) + "."

    print(prediction_string)
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

        evaluation_superset_label_ids = \
            superset_class_names_to_superset_class_ids(
                evaluation_set.superset_labels)

        if evaluation_set.excluded_superset_classes:
            excluded_superset_class_ids = \
                superset_class_names_to_superset_class_ids(
                    evaluation_set.excluded_superset_classes)
        else:
            excluded_superset_class_ids = []

    cluster_ids = predicted_labels = predicted_superset_labels = None

    if method == "copy":
        predicted_labels = evaluation_set.labels
        predicted_superset_labels = evaluation_set.superset_labels

    elif method == "k-means":

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

    elif method == "DBSCAN":

        minimum_neighbourhood_size = 2 * evaluation_set.number_of_features

        knn_model = NearestNeighbors(
            n_neighbors=minimum_neighbourhood_size - 1
        )
        knn_model.fit(evaluation_set.values)
        knn_distance_matrix, knn_index_matrix = knn_model.kneighbors(
            evaluation_set.values)
        knn_distances = knn_distance_matrix[:, -1]
        knn_distances_sorted = numpy.sort(knn_distances)[::-1]

        cut_off = int(0.5 * len(knn_distances))
        knee_locator = KneeLocator(
            x=numpy.arange(cut_off),
            y=knn_distances_sorted[:cut_off],
            # S=1.0,
            curve="convex",
            direction="decreasing"
        )
        index_knee = knee_locator.knee
        maximum_neighbour_distance = knn_distances_sorted[index_knee]

        model = DBSCAN(
            eps=maximum_neighbour_distance,
            min_samples=minimum_neighbourhood_size
        )
        cluster_ids = model.fit_predict(evaluation_set.values)

    else:
        raise ValueError("Prediction method not found: `{}`.".format(method))

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
            predicted_superset_labels = \
                superset_class_ids_to_superset_class_names(
                    predicted_superset_label_ids)

    prediction_duration = time() - prediction_time_start
    print("Labels predicted ({}).".format(
        format_duration(prediction_duration)))

    return cluster_ids, predicted_labels, predicted_superset_labels


def map_cluster_ids_to_label_ids(label_ids, cluster_ids,
                                 excluded_class_ids=[]):
    unique_cluster_ids = numpy.unique(cluster_ids).tolist()
    predicted_label_ids = numpy.zeros_like(cluster_ids)
    for unique_cluster_id in unique_cluster_ids:
        idx = cluster_ids == unique_cluster_id
        lab = label_ids[idx]
        for excluded_class_id in excluded_class_ids:
            lab = lab[lab != excluded_class_id]
        if len(lab) == 0:
            continue
        predicted_label_ids[idx] = scipy.stats.mode(lab)[0]
    return predicted_label_ids
