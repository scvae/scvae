from auxiliary import properString, formatDuration
from sklearn.cluster import KMeans

from time import time

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
    
    if prediction_method == "copy":
        predicted_labels = evaluation_set.labels
    
    elif prediction_method == "test":
        model = KMeans(n_clusters = number_of_classes, random_state = 0)
        model.fit(evaluation_set.values)
        predicted_labels = model.labels_
    
    elif prediction_method == "k-means":
        model = KMeans(n_clusters = number_of_classes, random_state = 0)
        model.fit(training_set.values)
        predicted_labels = model.predict(evaluation_set.values)
    
    else:
        raise ValueError("Prediction method not found: `{}`.".format(
            prediction_method)) 
    
    prediction_duration = time() - prediction_time_start
    print("Labels predicted ({}).".format(
        formatDuration(prediction_duration)))
    
    return predicted_labels
