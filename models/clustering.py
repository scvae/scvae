from auxiliary import normaliseString

def clusterDataSet(data_set, clustering_method = "copy"):
    
    clustering_method = normaliseString(clustering_method)
    
    if clustering_method == "copy":
        predicted_labels = data_set.labels
    else:
        raise ValueError("Clustering method not found: `{}`.".format(
            clustering_method)) 
    
    return predicted_labels
