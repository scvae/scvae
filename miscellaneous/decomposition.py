import numpy
import scipy
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE

from auxiliary import normaliseString, properString
from miscellaneous.incremental_pca import IncrementalPCA

DECOMPOSITION_METHOD_NAMES = {
    "PCA": ["pca"],
    "SVD": ["svd"],
    "ICA": ["ica"],
    "t-SNE": ["t_sne", "tsne"]
}

DECOMPOSITION_METHOD_LABEL = {
    "PCA": "PC",
    "SVD": "SVD",
    "ICA": "IC",
    "t-SNE": "tSNE"
}

DEFAULT_DECOMPOSITION_METHOD = "PCA"
DEFAULT_DECOMPOSITION_DIMENSIONALITY = 2

MAXIMUM_FEATURE_SIZE_FOR_NORMAL_PCA = 2000

def decompose(values, other_value_sets=[], centroids={}, method=None,
              number_of_components=None, random=False):
    
    # Setup
    
    ## Method
    
    if method is None:
        method = DEFAULT_DECOMPOSITION_METHOD
    
    method = properString(normaliseString(method), DECOMPOSITION_METHOD_NAMES)
    
    ## Number of components
    
    if number_of_components is None:
        number_of_components = DEFAULT_DECOMPOSITION_DIMENSIONALITY
    
    ## Other value sets
    
    if other_value_sets is not None \
        and not isinstance(other_value_sets, (list, tuple)):
        other_value_sets = [other_value_sets]
    
    ## Randomness
    
    if random:
        random_state = None
    else:
        random_state = 42
    
    # Method
    
    if method == "PCA":
        if values.shape[1] <= MAXIMUM_FEATURE_SIZE_FOR_NORMAL_PCA \
            and not scipy.sparse.issparse(values):
            model = PCA(n_components=number_of_components)
        else:
            model = IncrementalPCA(
                n_components=number_of_components,
                batch_size=100
            )
    elif method == "SVD":
        model = TruncatedSVD(n_components=number_of_components)
    elif method == "ICA":
        model = FastICA(n_components=number_of_components)
    elif method == "t-SNE":
        if number_of_components < 4:
            tsne_method = "barnes_hut"
        else:
            tsne_method = "exact"
        model = TSNE(
            n_components=number_of_components,
            method=tsne_method,
            random_state=random_state
        )
    else:
        raise ValueError("Method `{}` not found.".format(method))
    
    # Fit and transform
    
    values_decomposed = model.fit_transform(values)
    
    if other_value_sets and method != "t_sne":
        other_value_sets_decomposed = []
        for other_values in other_value_sets:
            other_value_decomposed = model.transform(other_values)
            other_value_sets_decomposed.append(other_value_decomposed)
    else:
        other_value_sets_decomposed = None
    
    if other_value_sets_decomposed and len(other_value_sets_decomposed) == 1:
        other_value_sets_decomposed = other_value_sets_decomposed[0]
    
    # Only supports centroids without data sets as top levels
    if centroids and method == "pca":
        if "means" in centroids:
            centroids = {"unknown": centroids}
        W = model.components_
        centroids_decomposed = {}
        for distribution, distribution_centroids in centroids.items():
            if distribution_centroids:
                centroids_distribution_decomposed = {}
                for parameter, values in distribution_centroids.items():
                    if parameter == "means":
                        shape = numpy.array(values.shape)
                        L = shape[-1]
                        reshaped_values = values.reshape(-1, L)
                        decomposed_values = model.transform(reshaped_values)
                        shape[-1] = number_of_components
                        new_values = decomposed_values.reshape(shape)
                    elif parameter == "covariance_matrices":
                        shape = numpy.array(values.shape)
                        L = shape[-1]
                        reshaped_values = values.reshape(-1, L, L)
                        B = reshaped_values.shape[0]
                        decomposed_values = numpy.empty((B, 2, 2))
                        for i in range(B):
                            decomposed_values[i] = W @ reshaped_values[i] @ W.T
                        shape[-2:] = number_of_components
                        new_values = decomposed_values.reshape(shape)
                    else:
                        new_values = values
                    centroids_distribution_decomposed[parameter] = new_values
                centroids_decomposed[distribution] = \
                    centroids_distribution_decomposed
            else:
                centroids_decomposed[distribution] = None
        if "unknown" in centroids_decomposed:
            centroids_decomposed = centroids_decomposed["unknown"]
    else:
        centroids_decomposed = None
    
    if other_value_sets != [] and centroids != {}:
        return values_decomposed, other_value_sets_decomposed, \
            centroids_decomposed
    elif other_value_sets != []:
        return values_decomposed, other_value_sets_decomposed
    elif centroids != {}:
        return values_decomposed, centroids_decomposed
    else:
        return values_decomposed
