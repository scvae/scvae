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


class SparseRowMatrix(scipy.sparse.csr_matrix):
    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        super().__init__(arg1, shape=shape, dtype=dtype, copy=copy)

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    def mean(self, axis=None):

        if axis is not None:
            return super().mean(axis)

        dtype = self.dtype.type

        if numpy.issubdtype(dtype, numpy.integer):
            dtype = numpy.float64

        self_sum = self.data.sum()
        self_mean = self_sum / self.size

        self_mean = self_mean.astype(dtype)

        return self_mean

    def std(self, axis=None, ddof=0):
        return numpy.sqrt(self.var(axis=axis, ddof=ddof))

    def var(self, axis=None, ddof=0):

        self_squared_mean = self.power(2).mean(axis)
        self_mean_squared = numpy.power(self.mean(axis), 2)

        var = self_squared_mean - self_mean_squared

        if ddof > 0:
            size = numpy.prod(self.shape)
            var = var * size / (size - ddof)

        return var


def sparsity(a, tolerance=1e-3, batch_size=None):

    def count_nonzero_values(b):
        return (b >= tolerance).sum()

    if scipy.sparse.issparse(a):
        size = numpy.prod(a.shape)
    else:
        size = a.size

    if batch_size:

        number_of_rows = a.shape[0]

        nonzero_count = 0

        for i in range(0, number_of_rows, batch_size):
            nonzero_count += count_nonzero_values(a[i:i+batch_size])

    else:
        nonzero_count = count_nonzero_values(a)

    a_sparsity = 1 - nonzero_count / size

    return a_sparsity
