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
"""The zero-inflated Poisson distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
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
dtype `dtype` and whose shape can be broadcast with `self.lambd`. `x` is only
legal if it is non-negative and its components are equal to integer values.
"""


class ZeroInflatedPoisson(distribution.Distribution):
  """Zero-inflated Poisson distribution.

  The zero-inflated Poisson distribution is parameterized by `lambd`, the rate parameter, and by 'pi', the zero-inflation parameter.

  The pmf of this distribution is:

  ```
  pmf(k) = pi + (1-pi) * e^(-lambd),        for k = 0
  pmf(k) = (1-pi) * e^(-lambd) * lambd^k / k!,   for k > 0
  ```

  """

  def __init__(self,
               lambd, 
               pi,
               validate_args=False,
               allow_nan_stats=True,
               name="ZeroInflatedPoisson"):
    """Construct zero-inflated Poisson distributions.

    Args:
      lambd: Floating point tensor, the rate parameter of the
        distribution(s). `lambd` must be positive.
      pi: Floating point tensor, the zero-inflation parameter of the
        distribution(s). `pi` must be in the interval [0, 1].
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `lambd > 0` as well as inputs to pmf computations are non-negative
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
    with ops.name_scope(name, values=[lambd, pi]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(lambd), check_ops.assert_positive(pi)] if
                                    validate_args else []):
        self._lambd = array_ops.identity(lambd, name="lambd")
        self._pi = array_ops.identity(pi, name="pi")
        contrib_tensor_util.assert_same_float_dtype((self._lambd, self._pi))
    super(ZeroInflatedPoisson, self).__init__(
        dtype=self._lambd.dtype,
        is_continuous=False,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._lambd, self._pi],
        name=ns)

  @property
  def lambd(self):
    """Rate parameter."""
    return self._lambd

  @property
  def pi(self):
    """Zero-inflation parameter."""
    return self._pi

  def _batch_shape(self):
    return array_ops.shape(self.lambd + self.pi)

  def _get_batch_shape(self):
    return common_shapes.broadcast_shape(
        self.lambd.get_shape(), self._pi.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  @distribution_util.AppendDocstring(_poisson_prob_note)
  def _log_prob(self, x):
    x = self._assert_valid_sample(x, check_integer=True)
    y_0 = math_ops.log(self.pi + (1 - self.pi) * math_ops.exp(-self.lambd))
    y_1 = math_ops.log(1 - self.pi) + x * math_ops.log(self.lambd) - self.lambd - math_ops.lgamma(x + 1)
    return tf.where(x > 0, y_1, y_0)

  @distribution_util.AppendDocstring(_poisson_prob_note)
  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  def _cdf(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    return math_ops.igammac(math_ops.floor(x + 1), self.lambd)

  def _mean(self):
    return (1-self.pi)*array_ops.identity(self.lambd)

  def _variance(self):
    return self.lambd * (1-self.pi)*(1+self.lambd*self.pi)

  def _std(self):
    return math_ops.sqrt(self.variance())

  @distribution_util.AppendDocstring(
      """Note that when `lambd` is an integer, there are actually two modes.
      Namely, `lambd` and `lambd - 1` are both modes. Here we return
      only the larger of the two modes.""")
  def _mode(self):
    return math_ops.floor(self.lambd)

  def _assert_valid_sample(self, x, check_integer=True):
    if not self.validate_args: return x
    with ops.name_scope("check_x", values=[x]):
      dependencies = [check_ops.assert_non_negative(x)]
      if check_integer:
        dependencies += [distribution_util.assert_integer_form(
            x, message="x has non-integer components.")]
      return control_flow_ops.with_dependencies(dependencies, x)
