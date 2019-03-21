# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

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

"""The Exponentially Modified Normal (Gaussian) distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow.python.framework import tensor_shape


__all__ = [
    "ExponentiallyModifiedNormal",
]


class ExponentiallyModifiedNormal(distribution.Distribution):
    """The exponentially modified normal (Gaussian) distribution.

    #### Mathematical details

    The probability density function (pdf) is,

    ```none
    pdf(x; mu, sigma, lambda) = exp(0.5 lambda (2 mu + lambda sigma**2 - 2 x))
                                * erfc((mu + lambda sigma**2 - x)
                                     / (sqrt(2) sigma))
                                / Z
    Z = 2 / lambda
    ```

    where

    * `loc = mu` is the mean,
    * `scale = sigma` is the std. deviation,
    * `rate = lambda`, and
    * `Z` is the normalisation constant.
    """

    def __init__(self,
                 loc,
                 scale,
                 rate,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="ExponentiallyModifiedNormal"):
        """Construct ExponentiallyModifiedNormal distributions.

        The parameters `loc`, `scale`, and `rate` must be shaped in a way that
        supports broadcasting (e.g. `loc + scale + rate` is a valid operation).

        Args:
            loc: Floating point tensor; the means of the normal component of
                the distribution(s).
            scale: Floating point tensor; the stddevs of the normal component
                 of the distribution(s). Must contain only positive values.
            rate: Floating point tensor; the rate of the exponential component
                 of the distribution(s). Must contain only positive values.
            validate_args: Python `bool`, default `False`. When `True`
                 distribution parameters are checked for validity despite
                 possibly degrading runtime performance. When `False` invalid
                 inputs may silently render incorrect outputs.
            allow_nan_stats: Python `bool`, default `True`. When `True`,
                statistics (e.g., mean, mode, variance) use the value "`NaN`"
                to indicate the result is undefined. When `False`, an
                exception is raised if one or more of the statistic's batch
                members are undefined.
            name: Python `str` name prefixed to Ops created by this class.

        Raises:
            TypeError: if `loc`, `scale`, and `rate` have different `dtype`.
        """
        parameters = dict(locals())
        with tf.name_scope(name, values=[loc, scale, rate]) as name:
            dtype = dtype_util.common_dtype([loc, scale, rate], tf.float32)
            loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
            scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
            rate = tf.convert_to_tensor(rate, name="rate", dtype=dtype)
            with tf.control_dependencies([
                tf.assert_positive(scale),
                tf.assert_positive(rate)
            ] if validate_args else []):
                self._loc = tf.identity(loc)
                self._scale = tf.identity(scale)
                self._rate = tf.identity(rate)
                tf.assert_same_float_dtype([
                    self._loc, self._scale, self._rate])
        super(ExponentiallyModifiedNormal, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._loc, self._scale, self._rate],
            name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(("loc", "scale", "rate"), ([tf.convert_to_tensor(
                sample_shape, dtype=tf.int32)] * 2)))

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for standard deviation."""
        return self._scale

    @property
    def rate(self):
        """Distribution parameter for the rate."""
        return self._rate

    def _batch_shape_tensor(self):
        tensors = [self.loc, self.scale, self.rate]
        return functools.reduce(tf.broadcast_dynamic_shape,
                                [tf.shape(tensor) for tensor in tensors])

    def _batch_shape(self):
        tensors = [self.loc, self.scale, self.rate]
        return functools.reduce(tf.broadcast_static_shape,
                                [tensor.shape for tensor in tensors])

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tensor_shape.scalar()

    def _sample_n(self, n, seed=None):
        # See
        # https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
        shape = tf.concat([[n], self.batch_shape_tensor()], 0)
        stream = seed_stream.SeedStream(
            seed, salt="exponentially_modified_normal")
        sampled_normal = tf.random_normal(
            shape=shape,
            mean=0.,
            stddev=1.,
            dtype=self.loc.dtype,
            seed=stream())
        sampled_uniform = tf.random_uniform(
            shape=shape,
            minval=np.finfo(self.dtype.as_numpy_dtype).tiny,
            maxval=1.,
            dtype=self.dtype,
            seed=stream())
        return (sampled_normal * self.scale + self.loc
                - tf.log(sampled_uniform) / self.rate)

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _cdf(self, x):
        u = self.rate * (x - self.loc)
        v = self.rate * self.scale
        v2 = tf.square(v)
        return (special_math.ndtr(u / v)
                - tf.exp(-u + 0.5 * v2
                         + tf.log(special_math.ndtr((u - v2) / v))))

    def _log_unnormalized_prob(self, x):
        u = self.rate * (x - self.loc)
        v = self.rate * self.scale
        v2 = tf.square(v)
        erfc_value = tf.clip_by_value(
            tf.erfc((-u + v2) / (math.sqrt(2.) * v)),
            np.finfo(self.dtype.as_numpy_dtype).tiny,
            np.inf)
        return -u + 0.5 * v2 + tf.log(erfc_value)

    def _log_normalization(self):
        return math.log(2.) - tf.log(self.rate)

    def _mean(self):
        return self.loc * tf.ones_like(self.scale) + 1 / self.rate

    def _variance(self):
        return (tf.square(self.scale) * tf.ones_like(self.loc)
                + tf.pow(self.rate, -2.))
