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

import numpy
import sklearn.metrics.cluster

CLUSTERING_METRICS = {}

MAXIMUM_NUMBER_OF_EXAMPLES_BEFORE_SAMPLING_SILHOUETTE_SCORE = 20000


def compute_clustering_metrics(evaluation_set):

    clustering_metric_values = {
        metric: {
            "clusters": None,
            "clusters; superset": None,
            "labels": None,
            "labels; superset": None
        }
        for metric in CLUSTERING_METRICS
    }

    for metric_name, metric_attributes in CLUSTERING_METRICS.items():

        metric_values = clustering_metric_values[metric_name]
        metric_kind = metric_attributes["kind"]
        metric_function = metric_attributes["function"]

        if metric_kind == "supervised":
            if evaluation_set.has_labels:
                if evaluation_set.has_predicted_cluster_ids:
                    metric_values["clusters"] = metric_function(
                        evaluation_set.labels,
                        evaluation_set.predicted_cluster_ids,
                        evaluation_set.excluded_classes
                    )
                if evaluation_set.has_predicted_labels:
                    metric_values["labels"] = metric_function(
                        evaluation_set.labels,
                        evaluation_set.predicted_labels,
                        evaluation_set.excluded_classes
                    )
            if evaluation_set.has_superset_labels:
                if evaluation_set.has_predicted_cluster_ids:
                    metric_values["clusters; superset"] = metric_function(
                        evaluation_set.superset_labels,
                        evaluation_set.predicted_cluster_ids,
                        evaluation_set.excluded_superset_classes
                    )
                if evaluation_set.has_predicted_superset_labels:
                    metric_values["labels; superset"] = metric_function(
                        evaluation_set.superset_labels,
                        evaluation_set.predicted_superset_labels,
                        evaluation_set.excluded_superset_classes
                    )
        elif metric_kind == "unsupervised":
            if evaluation_set.has_predicted_cluster_ids:
                metric_values["clusters"] = metric_function(
                    evaluation_set.values,
                    evaluation_set.predicted_cluster_ids
                )
            if evaluation_set.has_predicted_labels:
                metric_values["labels"] = metric_function(
                    evaluation_set.values,
                    evaluation_set.predicted_labels
                )
            if evaluation_set.has_predicted_superset_labels:
                metric_values["labels; superset"] = metric_function(
                    evaluation_set.values,
                    evaluation_set.predicted_superset_labels
                )

    return clustering_metric_values


def _register_clustering_metric(name, kind):
    def decorator(function):
        CLUSTERING_METRICS[name] = {
            "kind": kind,
            "function": function
        }
        return function
    return decorator


@_register_clustering_metric(name="adjusted Rand index", kind="supervised")
def adjusted_rand_index(labels, predicted_labels, excluded_classes=None):
    labels, predicted_labels = _exclude_classes_from_label_set(
        labels, predicted_labels, excluded_classes=excluded_classes)
    return sklearn.metrics.cluster.adjusted_rand_score(
        labels, predicted_labels)


@_register_clustering_metric(
    name="adjusted mutual information", kind="supervised")
def adjusted_mutual_information(labels, predicted_labels,
                                excluded_classes=None):
    labels, predicted_labels = _exclude_classes_from_label_set(
        labels, predicted_labels, excluded_classes=excluded_classes)
    return sklearn.metrics.cluster.adjusted_mutual_info_score(
        labels, predicted_labels, average_method="arithmetic")


@_register_clustering_metric(name="silhouette score", kind="unsupervised")
def silhouette_score(values, predicted_labels):
    number_of_predicted_classes = numpy.unique(predicted_labels).shape[0]
    number_of_examples = values.shape[0]

    if (number_of_predicted_classes < 2
            or number_of_predicted_classes > number_of_examples - 1):
        return numpy.nan

    sample_size = None

    if (number_of_examples
            > MAXIMUM_NUMBER_OF_EXAMPLES_BEFORE_SAMPLING_SILHOUETTE_SCORE):
        sample_size = (
            MAXIMUM_NUMBER_OF_EXAMPLES_BEFORE_SAMPLING_SILHOUETTE_SCORE)

    score = sklearn.metrics.silhouette_score(
        X=values,
        labels=predicted_labels,
        sample_size=sample_size
    )

    return score


def accuracy(labels, predicted_labels, excluded_classes=None):
    labels, predicted_labels = _exclude_classes_from_label_set(
        labels, predicted_labels, excluded_classes=excluded_classes)
    return numpy.mean(predicted_labels == labels)


def _exclude_classes_from_label_set(*label_sets, excluded_classes=None):

    if excluded_classes is None:
        excluded_classes = []

    labels = label_sets[0]
    other_label_sets = list(label_sets[1:])

    for excluded_class in excluded_classes:
        included_indices = labels != excluded_class
        labels = labels[included_indices]
        for i in range(len(other_label_sets)):
            other_label_sets[i] = other_label_sets[i][included_indices]

    if other_label_sets:
        return [labels] + other_label_sets
    else:
        return labels
