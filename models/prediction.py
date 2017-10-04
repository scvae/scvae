import numpy
import scipy.stats
from sklearn.cluster import KMeans

from time import time

from auxiliary import properString, formatDuration

prediction_method_names = {
    "k-means": ["k_means", "kmeans"],
    "test": ["test"],
    "copy": ["copy"]
}

def predictLabels(training_set, evaluation_set, prediction_method = "copy",
    number_of_classes = 1):
    
    prediction_method = properString(prediction_method,
        prediction_method_names)
    
    print("Predicting labels for evaluation set using {} on {} training set.".format(
        prediction_method, training_set.version))
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
    
    if prediction_method == "copy":
        predicted_labels = evaluation_set.labels
    
    elif prediction_method == "test":
        model = KMeans(n_clusters = number_of_classes, random_state = 0)
        model.fit(evaluation_set.values)
        predicted_labels = model.labels_
    
    elif prediction_method == "k-means":
        model = KMeans(n_clusters = number_of_classes, random_state = 0)
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
            predicted_labels = cluster_ids
    
    else:
        raise ValueError("Prediction method not found: `{}`.".format(
            prediction_method)) 
    
    prediction_duration = time() - prediction_time_start
    print("Labels predicted ({}).".format(
        formatDuration(prediction_duration)))
    
    return predicted_labels

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

def accuracy(labels, predicted_labels, excluded_classes = []):
    for excluded_class in excluded_classes:
        included_indices = labels != excluded_class
        labels = labels[included_indices]
        predicted_labels = predicted_labels[included_indices]
    return numpy.mean(predicted_labels == labels)
