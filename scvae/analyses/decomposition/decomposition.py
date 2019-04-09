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
import scipy
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE

from analyses.decomposition.incremental_pca import IncrementalPCA
from auxiliary import normalise_string, proper_string

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

    if method is None:
        method = DEFAULT_DECOMPOSITION_METHOD
    method = proper_string(
        normalise_string(method), DECOMPOSITION_METHOD_NAMES)

    if number_of_components is None:
        number_of_components = DEFAULT_DECOMPOSITION_DIMENSIONALITY

    if (other_value_sets is not None
            and not isinstance(other_value_sets, (list, tuple))):
        other_value_sets = [other_value_sets]

    if random:
        random_state = None
    else:
        random_state = 42

    if method == "PCA":
        if (values.shape[1] <= MAXIMUM_FEATURE_SIZE_FOR_NORMAL_PCA
                and not scipy.sparse.issparse(values)):
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
    if centroids and method == "PCA":
        if "means" in centroids:
            centroids = {"unknown": centroids}
        components = model.components_
        centroids_decomposed = {}
        for distribution, distribution_centroids in centroids.items():
            if distribution_centroids:
                centroids_distribution_decomposed = {}
                for parameter, parameter_values in (
                        distribution_centroids.items()):
                    if parameter == "means":
                        shape = numpy.array(parameter_values.shape)
                        original_dimension = shape[-1]
                        reshaped_parameter_values = parameter_values.reshape(
                            -1, original_dimension)
                        decomposed_parameter_values = model.transform(
                            reshaped_parameter_values)
                        shape[-1] = number_of_components
                        new_parameter_values = (
                            decomposed_parameter_values.reshape(shape))
                    elif parameter == "covariance_matrices":
                        shape = numpy.array(parameter_values.shape)
                        original_dimension = shape[-1]
                        reshaped_parameter_values = parameter_values.reshape(
                            -1, original_dimension, original_dimension)
                        n_centroids = reshaped_parameter_values.shape[0]
                        decomposed_parameter_values = numpy.empty(
                            shape=(n_centroids, 2, 2))
                        for i in range(n_centroids):
                            decomposed_parameter_values[i] = (
                                components
                                @ reshaped_parameter_values[i]
                                @ components.T
                            )
                        shape[-2:] = number_of_components
                        new_parameter_values = (
                            decomposed_parameter_values.reshape(shape))
                    else:
                        new_parameter_values = parameter_values
                    centroids_distribution_decomposed[parameter] = (
                        new_parameter_values)
                centroids_decomposed[distribution] = (
                    centroids_distribution_decomposed)
            else:
                centroids_decomposed[distribution] = None
        if "unknown" in centroids_decomposed:
            centroids_decomposed = centroids_decomposed["unknown"]
    else:
        centroids_decomposed = None

    if other_value_sets != [] and centroids != {}:
        return (
            values_decomposed,
            other_value_sets_decomposed,
            centroids_decomposed
        )
    elif other_value_sets != []:
        return values_decomposed, other_value_sets_decomposed
    elif centroids != {}:
        return values_decomposed, centroids_decomposed
    else:
        return values_decomposed
