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
"""The zero-inflated distribution class."""

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


class CountCategories(distribution.Distribution):
  """Zero-inflated distribution.

  The zero-inflated distribution is parameterized by `lambd`, the rate parameter, and by 'pi', the zero-inflation parameter.

  The pmf of this distribution is:

  ```
  pmf(k) = pi + (1-pi) * e^(-lambd),        for k = 0
  pmf(k) = (1-pi) * e^(-lambd) * lambd^k / k!,   for k > 0
  ```

  """

  def __init__(self, 
               dist,
               cat,
               validate_args=False,
               allow_nan_stats=True,
               name="CountCategories"):
    """Construct zero-inflated distributions.

    Args:
      dist: A 'Distribution' instance.
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

    if not isinstance(dist, distribution.Distribution):
      raise TypeError(
          "dist must be Distribution instance"
          " but saw: %s" % dist)

    if not isinstance(cat, categorical.Categorical):
      raise TypeError("cat must be a Categorical distribution, but saw: %s" % cat)
    is_continuous = dist.is_continuous

    static_event_shape = dist.get_event_shape()
    static_batch_shape = cat.get_batch_shape()
    static_batch_shape = static_batch_shape.merge_with(dist.get_batch_shape())

    if static_event_shape.ndims is None:
      raise ValueError(
          "Expected to know rank(event_shape) from distribution, but "
          "it does not provide a static number of ndims")

    with ops.name_scope(name, values=cat._graph_parents + dist._graph_parents) as ns:
      
      cat_batch_shape = cat.batch_shape()
      cat_batch_rank = array_ops.size(cat_batch_shape)
      if validate_args:
        batch_shapes = dist.batch_shape()
        batch_ranks = [array_ops.size(bs) for bs in batch_shapes]
        check_message = ("components[%d] batch shape must match cat "
                         "batch shape")
        self._assertions = [
            check_ops.assert_equal(
                cat_batch_rank, batch_ranks[di], message=check_message % di)
            for di in range(len(components))
        ]
        self._assertions += [
            check_ops.assert_equal(
                cat_batch_shape, batch_shapes[di], message=check_message % di)
            for di in range(len(components))
        ]
      else:
        self._assertions = []

      self._dist = dist
      self._cat = cat
      self._num_classes = cat.num_classes
      self._static_event_shape = static_event_shape
      self._static_batch_shape = static_batch_shape
    

    graph_parents = self._cat._graph_parents
    graph_parents += self._dist._graph_parents

    super(CountCategories, self).__init__(
        dtype=dist.dtype,
        is_continuous=is_continuous,
        is_reparameterized=False,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=graph_parents,
        name=ns)

  @property
  def dist(self):
    """Distribution."""
    return self._dist

  @property
  def cat(self):
    """Count Categories"""
    return self._cat

  @property
  def num_classes(self):
    """Scalar `int32` tensor: the number of classes."""
    return self._num_classes

  def _batch_shape(self):
    return self._cat.batch_shape()

  def _get_batch_shape(self):
    return self._static_batch_shape

  def _event_shape(self):
    return self._dist.event_shape()

  def _get_event_shape(self):
    return self._static_event_shape

  def _log_prob(self, x):
    with ops.control_dependencies(self._assertions):
      x = ops.convert_to_tensor(x, name='x')
      x = self._assert_valid_sample(x, check_integer=self._is_continuous)
      cat_log_prob = self._cat.log_prob(x)
      dist_log_prob = 
      return where(x < self.num_classes, self._cat.log_prob(x), self._cat.log_prob(tf.clip_by_value(x, 0, self.num_classes)) + self._dist.log_prob(x - self.num_classes))

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  # TODO: Implement the cumulative distribution function
  def _cdf(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    return math_ops.igammac(math_ops.floor(x + 1), self.lambd)


  def _mean(self):
    cat_mode = self._cat.mode()    
    return where(cat_mode < self.num_classes, cat_mode, self._cat.prob(self.num_classes) * (self._dist.mean() + self.num_classes))

  def _variance(self):
    # sigma^2 = mu + (pi/(1 - pi)) * mu^2
    mu = self._mean()
    return mu #(self._pi / (1 - self._pi)) * math_ops.square(mu)

  def _std(self):
    return math_ops.sqrt(self.variance())

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
