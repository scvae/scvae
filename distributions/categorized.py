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
"""The categorized class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import where
from tensorflow.contrib.distributions.python.ops import categorical
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
from tensorflow.python.ops import clip_ops


class Categorized(distribution.Distribution):
  """A categorized distribution.

  The categorized distribution is parameterized by the parameters in `dist` for the last bin in the `cat` distribution parameterized by k 'pi_k', the class probability parameter.

  The pmf of this distribution is:

  ```
  pmf(x = k)  = pi_k                      for x = k
  pmf(x >= K) = pi_K * pmf_D(x - K)       for x >= K
  ```
  where, k is \in [0, K], pmf_D(x-K) is the pmf of the distribution shifted by the K classes in the categorical. 
  """

  def __init__(self, 
               dist,
               cat,
               validate_args=False,
               allow_nan_stats=True,
               name="categorized"):
    """Construct categorized distributions.

    Args:
      dist: A 'Distribution' instance.
      cat: A `Categorical` instance.
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
      self._num_cat_classes = math_ops.cast(cat.num_classes-1, dtypes.float32)
      self._static_event_shape = static_event_shape
      self._static_batch_shape = static_batch_shape
    

    graph_parents = self._cat._graph_parents
    graph_parents += self._dist._graph_parents

    super(Categorized, self).__init__(
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
  def num_cat_classes(self):
    """K, Scalar `float32` tensor: the number of categorical classes without distribution."""
    return self._num_cat_classes

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
      cat_log_prob = self._cat.log_prob(math_ops.cast(clip_ops.clip_by_value(x, 0, self.num_cat_classes), dtypes.int32))
      #dist_log_prob = 
      return where(x < self.num_cat_classes, cat_log_prob, 
        cat_log_prob + self._dist.log_prob(x - self.num_cat_classes))

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  # TODO: Implement the cumulative distribution function
  def _cdf(self, x):
    x = self._assert_valid_sample(x, check_integer=False)
    return math_ops.igammac(math_ops.floor(x + 1), self.lambd)


  def _mean(self):
    cat_mode = math_ops.cast(self._cat.mode(), dtypes.float32)   
    return where(cat_mode < self.num_cat_classes, cat_mode, self._cat.prob(math_ops.cast(self.num_cat_classes,dtypes.int32)) * (self._dist.mean() + self.num_cat_classes))

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
