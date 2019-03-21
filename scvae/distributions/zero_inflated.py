# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

# ======================================================================== #
#
# Copyright (c) 2017 - 2019 scVAE authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================== #

"""The ZeroInflated distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import where
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import reparameterization


class ZeroInflated(distribution.Distribution):
    """zero-inflated distribution.

    The `zero-inflated` object implements batched zero-inflated distributions.
    The zero-inflated model is defined by a zero-inflation rate
    and a python list of `Distribution` objects.

    Methods supported include `log_prob`, `prob`, `mean`, `sample`, and
    `entropy_lower_bound`.
    """

    def __init__(self,
                 dist,
                 pi,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="ZeroInflated"):
        """Initialise a zero-inflated distribution.

        A `ZeroInflated` is defined by a zero-inflation rate (`pi`,
        representing the probabilities of excess zeroes) and a `Distribution`
        object having matching dtype, batch shape, event shape, and continuity
        properties (the dist).

        Args:
            pi: A zero-inflation rate, representing the probabilities of excess
                zeroes.
            dist: A `Distribution` instance.
                The instance must have `batch_shape` matching the
                zero-inflation rate.
            validate_args: Python `bool`, default `False`. If `True`, raise a
                runtime error if batch or event ranks are inconsistent between
                pi and any of the distributions. This is only checked if the
                ranks cannot be determined statically at graph construction
                time.
            allow_nan_stats: Boolean, default `True`. If `False`, raise an
                 exception if a statistic (e.g. mean/mode/etc...) is undefined
                for any batch member. If `True`, batch members with valid
                parameters leading to undefined statistics will return NaN for
                this statistic.
            name: A name for this distribution (optional).

        Raises:
            TypeError: If pi is not a zero-inflation rate, or `dist` is not
                `Distibution` are not instances of `Distribution`, or do not
                 have matching `dtype`.
            ValueError: If `dist` is an empty list or tuple, or its
                elements do not have a statically known event rank.
                If `pi.num_classes` cannot be inferred at graph creation time,
                or the constant value of `pi.num_classes` is not equal to
                `len(dist)`, or all `dist` and `pi` do not have
                matching static batch shapes, or all dist do not
                have matching static event shapes.
        """
        parameters = locals()
        if not dist:
            raise ValueError("dist must be non-empty")

        if not isinstance(dist, distribution.Distribution):
            raise TypeError(
                "dist must be a Distribution instance"
                " but saw: %s" % dist)

        dtype = dist.dtype
        static_event_shape = dist.event_shape
        static_batch_shape = pi.get_shape()

        if static_event_shape.ndims is None:
            raise ValueError(
                "Expected to know rank(event_shape) from dist, but "
                "the distribution does not provide a static number of ndims")

        # Ensure that all batch and event ndims are consistent.
        with ops.name_scope(name, values=[pi]):
            with ops.control_dependencies([check_ops.assert_positive(pi)] if
                                          validate_args else []):
                pi_batch_shape = array_ops.shape(pi)
                pi_batch_rank = array_ops.size(pi_batch_shape)
                if validate_args:
                    dist_batch_shape = dist.batch_shape_tensor()
                    dist_batch_rank = array_ops.size(dist_batch_shape)
                    check_message = (
                        "dist batch shape must match pi batch shape")
                    self._assertions = [check_ops.assert_equal(
                        pi_batch_rank, dist_batch_rank, message=check_message)]
                    self._assertions += [
                        check_ops.assert_equal(
                            pi_batch_shape, dist_batch_shape,
                            message=check_message)]
                else:
                    self._assertions = []

                self._pi = pi
                self._dist = dist
                self._static_event_shape = static_event_shape
                self._static_batch_shape = static_batch_shape

        # We let the zero-inflated distribution access _graph_parents since its
        # arguably more like a baseclass.
        graph_parents = [self._pi]
        graph_parents += self._dist._graph_parents

        super(ZeroInflated, self).__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                graph_parents=graph_parents,
                name=name)

    @property
    def pi(self):
        return self._pi

    @property
    def dist(self):
        return self._dist

    def _batch_shape_tensor(self):
        return array_ops.shape(self._pi)

    def _batch_shape(self):
        return self._static_batch_shape

    def _event_shape_tensor(self):
        return self._dist.event_shape_tensor()

    def _event_shape(self):
        return self._static_event_shape

    def _mean(self):
        with ops.control_dependencies(self._assertions):
            # These should all be the same shape by virtue of matching
            # batch_shape and event_shape.
            return (1-self._pi) * self._dist.mean()

    def _variance(self):
        with ops.control_dependencies(self._assertions):
            # These should all be the same shape by virtue of matching
            # batch_shape and event_shape.
            return ((1-self._pi) * (self._dist.variance()
                                    + math_ops.square(self._dist.mean()))
                    - math_ops.square(self._mean()))

    def _log_prob(self, x):
        with ops.control_dependencies(self._assertions):
            x = ops.convert_to_tensor(x, name="x")
            y_0 = math_ops.log(self.pi + (1 - self.pi) * self._dist.prob(x))
            y_1 = math_ops.log(1 - self.pi) + self._dist.log_prob(x)
            return where(x > 0, y_1, y_0)

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))
