# ======================================================================== #
#
# Copyright (c) 2017 - 2020 scVAE authors
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
import scipy.sparse

from sklearn.decomposition import IncrementalPCA as SKLIncrementalPCA
from sklearn.utils import check_array, gen_batches
from sklearn.utils.validation import check_is_fitted


class IncrementalPCA(SKLIncrementalPCA):
    """Incremental PCA supporting large sparse matrices."""
    def __init__(self, n_components=None, whiten=False, copy=True,
                 batch_size=None):
        super(IncrementalPCA, self).__init__(
            n_components=n_components,
            whiten=whiten,
            copy=copy,
            batch_size=batch_size
        )

    def fit(self, x, y=None):

        self.components_ = None
        self.n_samples_seen_ = 0
        self.mean_ = .0
        self.var_ = .0
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.noise_variance_ = None

        x = check_array(
            x, accept_sparse=["csr", "csc"], copy=self.copy,
            dtype=[numpy.float64, numpy.float32])
        n_samples, n_features = x.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(n_samples, self.batch_size_,
                                 min_batch_size=self.n_components or 0):
            self.partial_fit(x[batch], check_input=False)

        return self

    def partial_fit(self, x, y=None, check_input=True):
        if check_input:
            x = check_array(
                x, accept_sparse=["csr", "csc"], copy=self.copy,
                dtype=[numpy.float64, numpy.float32])

        if scipy.sparse.issparse(x):
            x = x.A

        return super(IncrementalPCA, self).partial_fit(
            x, y=y, check_input=check_input)

    def transform(self, x):
        check_is_fitted(self, ["mean_", "components_"], all_or_any=all)

        x = check_array(x, accept_sparse=["csr", "csc"])
        n_samples, n_features = x.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        x_transformed = numpy.empty([n_samples, self.n_components])

        for batch in gen_batches(n_samples, self.batch_size_):
            x_transformed[batch] = self.partial_transform(
                x[batch], check_input=False)

        return x_transformed

    def partial_transform(self, x, check_input=True):
        check_is_fitted(self, ["mean_", "components_"], all_or_any=all)

        x = check_array(x, accept_sparse=["csr", "csc"])

        if self.mean_ is not None:
            x = x - self.mean_
        x_transformed = numpy.dot(x, self.components_.T)

        if self.whiten:
            x_transformed /= numpy.sqrt(self.explained_variance_)

        return x_transformed
