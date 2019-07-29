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
import sklearn.metrics


def correlation_matrix(data_matrix, axis=None):
    if axis in [None, 0, "examples", "rows"]:
        pass
    elif axis in [1, "features", "columns"]:
        data_matrix = data_matrix.T
    correlation_matrix = 1 - sklearn.metrics.pairwise_distances(
        # The values are transposed to get correlations for the
        # features and not the examples
        data_matrix, metric="correlation")
    return correlation_matrix


def most_correlated_variable_pairs_from_correlation_matrix(correlation_matrix,
                                                           n_limit=None):

    n_features = correlation_matrix.shape[0]
    n_feature_pairs = int(n_features * (n_features - 1) / 2)

    absolute_correlations_off_diagonal = numpy.ma.masked_array(
        numpy.absolute(correlation_matrix), mask=numpy.tri(n_features))
    sorted_indices = numpy.unravel_index(
        absolute_correlations_off_diagonal.argsort(
            axis=None, endwith=False),
        shape=correlation_matrix.shape)
    sorted_feature_pair_indices = [
        tuple(index_pair) for index_pair in numpy.array(sorted_indices).T]

    if n_limit is None:
        n_limit = n_feature_pairs
    else:
        n_limit = min(n_limit, n_feature_pairs)

    largest_correlations_feature_pairs = sorted_feature_pair_indices[-n_limit:]

    return largest_correlations_feature_pairs
