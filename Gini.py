import numpy as np

def gini_coefficients(array, eps=1e-8):
    """Calculate the Gini coefficients along last axis of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    # Array should be normalized frequencies along last dim.
    array = array.T / np.sum(array.T,axis=0)
    array = array.T

    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array = np.sort(array, axis=-1) #values must be sorted
    
    array = np.clip(array, eps, array) #values cannot be 0
    
    n = array.shape[-1] # number of array elements
    index_vector =  2 * np.arange(1,n+1) - n - 1  # 1-indexing vector for each array element.

    return np.dot(array, index_vector) / n * np.sum(array, axis=-1) #Gini coefficients along the 1st dimension