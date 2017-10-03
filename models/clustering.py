from auxiliary import properString, formatDuration
from sklearn.cluster import KMeans

from time import time

clustering_method_names = {
    "k-means": ["k_means", "kmeans"],
    "copy": ["copy"]
}

def clusterDataSet(data_set, clustering_method = "copy", number_of_clusters = 1):
    
    clustering_method = properString(clustering_method,
        clustering_method_names)
    
    print("Clustering examples in {} {} set using {}.".format(
        data_set.version,
        data_set.kind,
        clustering_method
    ))
    clustering_time_start = time()
    
    if clustering_method == "copy":
        predicted_labels = data_set.labels
    
    if clustering_method == "k-means":
        model = KMeans(n_clusters = number_of_clusters, random_state = 0)
        model.fit(data_set.values)
        predicted_labels = model.labels_
    
    else:
        raise ValueError("Clustering method not found: `{}`.".format(
            clustering_method)) 
    
    clustering_duration = time() - clustering_time_start
    print("Examples clustered ({}).".format(
        formatDuration(clustering_duration)))
    
    return predicted_labels
