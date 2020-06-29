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

import tensorflow as tf
from tensorflow_probability import distributions as tfd


class MultivariateNormalDiag(tfd.MultivariateNormalDiag):
    def __init__(self,
                 loc=None,
                 scale_diag=None,
                 scale_identity_multiplier=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='MultivariateNormalDiag'):
        if loc is not None:
            loc = tf.expand_dims(loc, axis=-2)
        if scale_diag is not None:
            scale_diag = tf.expand_dims(scale_diag, axis=-2)
        if scale_identity_multiplier is not None:
            scale_identity_multiplier = tf.expand_dims(
                scale_identity_multiplier, axis=-2)
        super().__init__(
            loc=loc,
            scale_diag=scale_diag,
            scale_identity_multiplier=scale_identity_multiplier,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name
        )

    def cdf(self, value):
        return super().cdf(tf.expand_dims(value, axis=-2))

    def covariance(self):
        return tf.squeeze(super().covariance(), axis=-3)

    def log_cdf(self, value):
        return super().log_cdf(tf.expand_dims(value, axis=-2))

    def log_prob(self, value):
        return super().log_prob(tf.expand_dims(value, axis=-2))

    def log_survival_function(self, value):
        return super().log_survival_function(tf.expand_dims(value, axis=-2))

    def mean(self):
        return tf.squeeze(super().mean(), axis=-2)

    def mode(self):
        return tf.squeeze(super().mode(), axis=-2)

    def prob(self, value):
        return super().prob(tf.expand_dims(value, axis=-2))

    def quantile(self, value):
        return super().quantile(tf.expand_dims(value, axis=-2))

    def sample(self, sample_shape=(), seed=None):
        return tf.squeeze(super().sample(
            sample_shape=sample_shape,
            seed=seed
        ), axis=-2)

    def stddev(self):
        return tf.squeeze(super().stddev(), axis=-2)

    def survival_function(self, value):
        return super().survival_function(tf.expand_dims(value, axis=-2))

    def variance(self):
        return tf.squeeze(super().variance(), axis=-2)


class MultivariateNormalTriL(tfd.MultivariateNormalTriL):
    def __init__(self,
                 loc=None,
                 scale_tril=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='MultivariateNormalTriL'):
        if loc is not None:
            loc = tf.expand_dims(loc, axis=-2)
        if scale_tril is not None:
            scale_tril = tf.expand_dims(scale_tril, axis=-3)
        super().__init__(
            loc=loc,
            scale_tril=scale_tril,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name
        )

    def cdf(self, value):
        return super().cdf(tf.expand_dims(value, axis=-2))

    def covariance(self):
        return tf.squeeze(super().covariance(), axis=-3)

    def log_cdf(self, value):
        return super().log_cdf(tf.expand_dims(value, axis=-2))

    def log_prob(self, value):
        return super().log_prob(tf.expand_dims(value, axis=-2))

    def log_survival_function(self, value):
        return super().log_survival_function(tf.expand_dims(value, axis=-2))

    def mean(self):
        return tf.squeeze(super().mean(), axis=-2)

    def mode(self):
        return tf.squeeze(super().mode(), axis=-2)

    def prob(self, value):
        return super().prob(tf.expand_dims(value, axis=-2))

    def quantile(self, value):
        return super().quantile(tf.expand_dims(value, axis=-2))

    def sample(self, sample_shape=(), seed=None):
        return tf.squeeze(super().sample(
            sample_shape=sample_shape,
            seed=seed
        ), axis=-2)

    def stddev(self):
        return tf.squeeze(super().stddev(), axis=-2)

    def survival_function(self, value):
        return super().survival_function(tf.expand_dims(value, axis=-2))

    def variance(self):
        return tf.squeeze(super().variance(), axis=-2)


class MultivariateNormalFullCovariance(tfd.MultivariateNormalFullCovariance):
    def __init__(self,
                 loc=None,
                 covariance_matrix=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='MultivariateNormalFullCovariance'):
        if loc is not None:
            loc = tf.expand_dims(loc, axis=-2)
        if covariance_matrix is not None:
            covariance_matrix = tf.expand_dims(covariance_matrix, axis=-3)
        super().__init__(
            loc=loc,
            covariance_matrix=covariance_matrix,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name
        )

    def cdf(self, value):
        return super().cdf(tf.expand_dims(value, axis=-2))

    def covariance(self):
        return tf.squeeze(super().covariance(), axis=-3)

    def log_cdf(self, value):
        return super().log_cdf(tf.expand_dims(value, axis=-2))

    def log_prob(self, value):
        return super().log_prob(tf.expand_dims(value, axis=-2))

    def log_survival_function(self, value):
        return super().log_survival_function(tf.expand_dims(value, axis=-2))

    def mean(self):
        return tf.squeeze(super().mean(), axis=-2)

    def mode(self):
        return tf.squeeze(super().mode(), axis=-2)

    def prob(self, value):
        return super().prob(tf.expand_dims(value, axis=-2))

    def quantile(self, value):
        return super().quantile(tf.expand_dims(value, axis=-2))

    def sample(self, sample_shape=(), seed=None):
        return tf.squeeze(super().sample(
            sample_shape=sample_shape,
            seed=seed
        ), axis=-2)

    def stddev(self):
        return tf.squeeze(super().stddev(), axis=-2)

    def survival_function(self, value):
        return super().survival_function(tf.expand_dims(value, axis=-2))

    def variance(self):
        return tf.squeeze(super().variance(), axis=-2)
