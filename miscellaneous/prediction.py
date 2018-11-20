# ======================================================================== #
# 
# Copyright (c) 2017 - 2018 scVAE authors
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
import scipy.stats
from sklearn.cluster import KMeans, MiniBatchKMeans

from time import time

from auxiliary import properString, formatDuration

PREDICTION_METHOD_NAMES = {
    "k-means": ["k_means", "kmeans"],
    "model": ["model"],
    "test": ["test"],
    "copy": ["copy"]
}

MAXIMUM_SAMPLE_SIZE_FOR_NORMAL_KMEANS = 10000

def predict(training_set, evaluation_set, method = "copy",
    number_of_classes = 1):
    
    method = properString(method, PREDICTION_METHOD_NAMES)
    
    print(
        "Predicting labels for evaluation set using {}".format(method),
        "on {} prediction training set.".format(training_set.version)
    )
    prediction_time_start = time()
    
    if evaluation_set.has_labels:
        
        class_names_to_class_ids = numpy.vectorize(lambda class_name:
            evaluation_set.class_name_to_class_id[class_name])
        class_ids_to_class_names = numpy.vectorize(lambda class_id:
            evaluation_set.class_id_to_class_name[class_id])
            
        evaluation_label_ids = class_names_to_class_ids(
            evaluation_set.labels)
        
        if evaluation_set.excluded_classes:
            excluded_class_ids = class_names_to_class_ids(
                evaluation_set.excluded_classes)
        else:
            excluded_class_ids = []
    
    if evaluation_set.has_superset_labels:
        
        superset_class_names_to_superset_class_ids = numpy.vectorize(
            lambda superset_class_name: evaluation_set\
                .superset_class_name_to_superset_class_id[superset_class_name]
        )
        superset_class_ids_to_superset_class_names = numpy.vectorize(
            lambda superset_class_id: evaluation_set\
                .superset_class_id_to_superset_class_name[superset_class_id]
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
    
    if method == "copy":
        cluster_ids = None
        predicted_labels = evaluation_set.labels
        predicted_superset_labels = evaluation_set.superset_labels
    
    elif method == "k-means":
        
        if training_set.number_of_examples \
            <= MAXIMUM_SAMPLE_SIZE_FOR_NORMAL_KMEANS:
                model = KMeans(
                    n_clusters=number_of_classes,
                    random_state=None
                )
        else:
            model = MiniBatchKMeans(
                n_clusters=number_of_classes,
                random_state=None,
                batch_size=100
            )
        
        model.fit(training_set.values)
        cluster_ids = model.predict(evaluation_set.values)
        
        if evaluation_set.has_labels:
            predicted_label_ids = mapClusterIDsToLabelIDs(
                evaluation_label_ids,
                cluster_ids,
                excluded_class_ids
            )
            predicted_labels = class_ids_to_class_names(predicted_label_ids)
        else:
            predicted_labels = None
        
        if evaluation_set.has_superset_labels:
            predicted_superset_label_ids = mapClusterIDsToLabelIDs(
                evaluation_superset_label_ids,
                cluster_ids,
                excluded_superset_class_ids
            )
            predicted_superset_labels = \
                superset_class_ids_to_superset_class_names(
                    predicted_superset_label_ids)
        else:
            predicted_superset_labels = None
    
    else:
        raise ValueError("Prediction method not found: `{}`.".format(method))
    
    prediction_duration = time() - prediction_time_start
    print("Labels predicted ({}).".format(
        formatDuration(prediction_duration)))
    
    return cluster_ids, predicted_labels, predicted_superset_labels

def mapClusterIDsToLabelIDs(label_ids, cluster_ids, excluded_class_ids = []):
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
