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
# ==============================================================================
"""The negative binomial distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import where
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


_poisson_prob_note = """
Note thet the input value must be a non-negative floating point tensor with
dtype `dtype` and whose shape can be broadcast with `self.p`. `x` is only
legal if it is non-negative and its components are equal to integer values.
"""


class NegativeBinomial(distribution.Distribution):
  """negative binomial distribution.

  The negative binomial distribution is parameterized by `p`, the rate parameter, and by 'r', the zero-inflation parameter.

  The pmf of this distribution is:

  ```
  pmf(k) = r + (1-r) * e^(-p),        for k = 0
  pmf(k) = (1-r) * e^(-p) * p^k / k!,   for k > 0
  ```

  """

  def __init__(self,
               r, 
               p,
               validate_args=False,
               allow_nan_stats=True,
               name="NegativeBinomial"):
    """Construct negative binomial distributions.

    Args:
      r: Floating point tensor, the number of failures before stop parameter of the distribution(s). `r` must be positive.
      p: Floating point tensor, the succes probability parameter of the distribution(s). `p` must be in the interval [0, 1].
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `p > 0` as well as inputs to pmf computations are non-negative
        integers. If validate_args is `False`, then `pmf` computations might
        return `NaN`, but can be evaluated at any real value.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[r, p]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(r), check_ops.assert_positive(p)] if
                                    validate_args else []):
        self._r = array_ops.identity(r, name="r")
        self._p = array_ops.identity(p, name="p")
        contrib_tensor_util.assert_same_float_dtype((self._r, self._p))
    super(NegativeBinomial, self).__init__(
        dtype=self._r.dtype,
        is_continuous=True,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._r, self._p],
        name=ns)


  @property
  def r(self):
    """Stopping-time parameter."""
    return self._r

  @property
  def p(self):
    """Succes probability parameter."""
    return self._p

  def _batch_shape(self):
    return array_ops.shape(self.p + self.r)

  def _get_batch_shape(self):
    return common_shapes.broadcast_shape(
        self.p.get_shape(), self._r.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _log_prob(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    y = math_ops.lgamma(x + self.r) - math_ops.lgamma(x + 1) - math_ops.lgamma(self.r) + x * math_ops.log(self.p) + self.r * math_ops.log(1 - self.p)
    return y

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  def _cdf(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    return math_ops.igammac(math_ops.floor(x + 1), self.p)

  def _mean(self):
    return (self.p * self.r) / (1 - self.p)

  def _variance(self):
    return self.p * self.r / math_ops.square(1-self.p)

  def _std(self):
    return math_ops.sqrt(self.variance())

  def _mode(self):
    return where(self.r > 1, math_ops.floor(self.p*(self.r-1)/(1-self.p)), 0.0)

  def _assert_valid_sample(self, x, check_integer=False):
    if not self.validate_args: return x
    with ops.name_scope("check_x", values=[x]):
      dependencies = [check_ops.assert_non_negative(x)]
      if check_integer:
        dependencies += [distribution_util.assert_integer_form(
            x, message="x has non-integer components.")]
      return control_flow_ops.with_dependencies(dependencies, x)
