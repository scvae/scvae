# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==========================================================================

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

"""The Lomax distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import reparameterization


class Lomax(distribution.Distribution):
    """Lomax distribution.

    The Lomax distribution is defined over positive real numbers and zero using
    parameters `concentration` (aka "alpha") and `scale` (aka "lambda").

    #### Mathematical Details

    The probability density function (pdf) is,

    ```none
    pdf(x; alpha, lambda, x >= 0) = (1 + x / lambda)**(-(alpha + 1)) / Z
    Z = lambda / alpha
    ```

    where:

    * `concentration = alpha`, `alpha > 0`,
    * `scale = lambda`, `lambda > 0`, and,
    * `Z` is the normalising constant.

    The cumulative density function (cdf) is,

    ```none
    cdf(x; alpha, lambda, x >= 0) = 1 - (1 + x / lambda)**(-alpha)
    ```

    Distribution parameters are automatically broadcast in all functions; see
    examples for details.

    #### Examples

    ```python
    dist = Lomax(concentration=3.0, scale=2.0)
    dist2 = Lomax(concentration=[3.0, 4.0], scale=[2.0, 3.0])
    ```

    """

    def __init__(self,
                 concentration,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="Lomax"):
        """Construct Lomax with `concentration` and `scale` parameters.

        The parameters `concentration` and `scale` must be shaped in a way that
        supports broadcasting (e.g. `concentration + scale` is a valid
        operation).

        Args:
            concentration: Floating point tensor, the concentration params of
                the distribution(s). Must contain only positive values.
            scale: Floating point tensor, the inverse scale params of the
                distribution(s). Must contain only positive values.
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
            TypeError: if `concentration` and `scale` are different dtypes.
        """
        parameters = locals()
        with ops.name_scope(name, values=[concentration, scale]):
            with ops.control_dependencies([
                check_ops.assert_positive(concentration),
                check_ops.assert_positive(scale),
            ] if validate_args else []):
                self._concentration = array_ops.identity(
                    concentration, name="concentration")
                self._scale = array_ops.identity(scale, name="scale")
                check_ops.assert_same_float_dtype(
                    [self._concentration, self._scale])
        super(Lomax, self).__init__(
            dtype=self._concentration.dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            parameters=parameters,
            graph_parents=[self._concentration,
                           self._scale],
            name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(("concentration", "scale"), ([ops.convert_to_tensor(
                sample_shape, dtype=dtypes.int32)] * 2)))

    @property
    def concentration(self):
        """Concentration parameter."""
        return self._concentration

    @property
    def scale(self):
        """Scale parameter."""
        return self._scale

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
            array_ops.shape(self.concentration),
            array_ops.shape(self.scale))

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
            self.concentration.get_shape(),
            self.scale.get_shape())

    def _event_shape_tensor(self):
        return constant_op.constant([], dtype=dtypes.int32)

    def _event_shape(self):
        return tensor_shape.scalar()

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))

    def _log_cdf(self, x):
        return math_ops.log(self._cdf(x))

    def _cdf(self, x):
        x = self._maybe_assert_valid_sample(x)
        return 1 - math_ops.pow(1 + x / self.scale, - self.concentration)

    def _log_unnormalized_prob(self, x):
        x = self._maybe_assert_valid_sample(x)
        return - (self.concentration + 1.) * math_ops.log(1 + x / self.scale)

    def _log_normalization(self):
        return math_ops.log(self.scale) - math_ops.log(self.concentration)

    @distribution_util.AppendDocstring(
        """The mean of a Lomax distribution is only defined for
        `concentration > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is
        `False`, an exception will be raised rather than returning `NaN`.""")
    def _mean(self):
        mean = self.scale / (self.concentration - 1)
        if self.allow_nan_stats:
            nan = array_ops.fill(
                self.batch_shape_tensor(),
                np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
                name="nan")
            return array_ops.where(self.concentration > 1., mean, nan)
        else:
            return control_flow_ops.with_dependencies([
                check_ops.assert_less(
                    array_ops.ones([], self.dtype),
                    self.concentration,
                    message="mean not defined when any concentration <= 1"
                ),
            ], mean)

    @distribution_util.AppendDocstring(
        """The variance of a Lomax distribution is only defined for
        `concentration > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is
        `False`, an exception will be raised rather than returning `NaN`.""")
    def _variance(self):
        variance = (math_ops.square(self.scale) * (self.concentration - 1)
                    / (math_ops.square(self.concentration - 1)
                    * (self.concentration - 2)))
        if self.allow_nan_stats:
            nan = array_ops.fill(
                    self.batch_shape_tensor(),
                    np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
                    name="nan")
            inf = array_ops.fill(
                    self.batch_shape_tensor(),
                    np.array(np.inf, dtype=self.dtype.as_numpy_dtype()),
                    name="inf")
            return array_ops.where(
                self.concentration > 2., variance, array_ops.where(
                    self.concentration > 1., inf, nan))
        else:
            return control_flow_ops.with_dependencies([
                check_ops.assert_less(
                    array_ops.ones([], self.dtype),
                    self.concentration,
                    message="variance not defined when any concentration <= 1"
                ),
            ], variance)

    def _stddev(self):
        return math_ops.sqrt(self._variance())

    def _mode(self):
        return array_ops.fill(
            self.batch_shape_tensor(),
            np.array(0, dtype=self.dtype.as_numpy_dtype()),
            name="zero")

    def _maybe_assert_valid_sample(self, x):
        check_ops.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
        if not self.validate_args:
            return x
        return control_flow_ops.with_dependencies([
                check_ops.assert_non_negative(x),
        ], x)
