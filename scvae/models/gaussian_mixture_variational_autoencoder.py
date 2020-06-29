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

import copy
import os
import shutil
from time import time

import numpy
import scipy.sparse
import tensorflow as tf
import tensorflow_probability as tfp

from scvae.analyses.metrics.clustering import accuracy
from scvae.analyses.prediction import map_cluster_ids_to_label_ids
from scvae.data.data_set import DataSet
from scvae.defaults import defaults
from scvae.distributions import (
    DISTRIBUTIONS, GAUSSIAN_MIXTURE_DISTRIBUTIONS, parse_distribution,
    Categorised)
from scvae.analyses.prediction import PredictionSpecifications
from scvae.models.utilities import (
    dense_layer, dense_layers,
    build_training_string, build_data_string,
    load_learning_curves, early_stopping_status,
    generate_unique_run_id_for_model, check_run_id,
    correct_model_checkpoint_path, remove_old_checkpoints,
    copy_model_directory, clear_log_directory,
    parse_numbers_of_samples, validate_model_parameters,
    batch_indices_for_subset)
from scvae.utilities import (
    format_duration, format_time,
    normalise_string, capitalise_string)


class GaussianMixtureVariationalAutoencoder:
    """Gaussian-mixture variational auto-encoder class.

    Arguments:
        feature_size (int): The number of features/genes in the data
            set to model.
        latent_size (int): The number of dimensions to use for the
            latent space.
        hidden_sizes (list(int)): A list of the number of units in each
            hidden layer of both the inference (encoder) and the
            generative (decoder) networks. The number of layers in each
            network is thus the length of this list.
            For the inference network, the order of the hidden layers
            is the same as for the list, while for the generative
            network, it is the reverse.
        reconstruction_distribution (str, optional): The name of the
            reconstruction distribution (or likelihood function; see
            :ref:`Training a model`)
        number_of_reconstruction_classes (int, optional): The number of
            counts to model directly, starting from zero (see
            :ref:`Training a model`).
        latent_distribution (str, optional): The name of the
            latent prior distribution: ``"gaussian_mixture"`` or
            ``"full_covariance_gaussian_mixture"`` (see
            :ref:`Training a model`).
        prior_probabilities_method (str, optional): Method for how to
            set the mixture coefficients for the latent prior
            distribution: ``"uniform"`` distribution, ``"custom"``
            (provide probabilities to ``prior_probabilities``), or
            ``"learn"`` during training.
        prior_probabilities (1-d array-like, optional): Prior
            probabilities required when ``prior_probabilities_method``
            is ``"custom"``.
        number_of_latent_clusters (int, optional): The number of latent
            clusters, which is also the number of components in the
            Gaussian-mixture model.
        minibatch_normalisation (bool, optional): If ``True``,
            normalise each random minibatch of data when training or
            evaluating the model.
        batch_correction (bool, optional): If ``True``, and if batches
            are present in data set to model, perform batch correction.
        number_of_batches (int, optional): The number of batches in the
            data set to model. Required, if ``batch_correction`` is
            ``True``.
        number_of_warm_up_epochs (int, optional): The number of epochs
            during the start of training with a linear weight on the KL
            divergence. This weight is gradually increased linearly
            from 0 to 1 for this number of epochs.
        log_directory (str, optional): Directory where model is saved.

    Attributes:
        feature_size: The number of features/genes which can be
            modelled.
        latent_size: The number of dimensions of the latent space.
        hidden_sizes:  A list of the number of units in each hidden
            layer of both the inference (encoder) and the generative
            (decoder) networks. The number of layers in each network is
            thus the length of this list. For the inference network,
            the order of the hidden layers is the same as for the list,
            while for the generative network, it is the reverse.
        reconstruction_distribution: An instance of the reconstruction
            distribution (or likelihood function) class used by the
            model.
        number_of_reconstruction_classes: The number of counts modelled
            directly, starting from zero.
        latent_distribution: An instance of the latent prior
            distribution class used by the model.
        prior_probabilities_method: Method for how the mixture
            coefficients for the latent prior distribution are set:
            ``"uniform"`` distribution, ``"custom"`` (given by
            ``prior_probabilities``), or ``"learn"`` during training.
        prior_probabilities: Prior probabilities when
            ``prior_probabilities_method`` is ``"custom"``.
        minibatch_normalisation: If ``True``, normalise each random
            minibatch of data when training or evaluating the model.
        batch_correction: If ``True``, and if batches are present in
            data set to model, perform batch correction.
        number_of_batches: The number of batches in the data set to
            model, when ``batch_correction`` is ``True``.
        number_of_warm_up_epochs: The number of epochs during the start
            of training with a linear weight on the KL divergence. This
            weight is gradually increased linearly from 0 to 1 for this
            number of epochs.
    """

    def __init__(self, feature_size, latent_size=None, hidden_sizes=None,
                 reconstruction_distribution=None,
                 number_of_reconstruction_classes=None,
                 latent_distribution=None,
                 prior_probabilities_method=None,
                 prior_probabilities=None,
                 number_of_latent_clusters=None,
                 minibatch_normalisation=None,
                 batch_correction=None,
                 number_of_batches=None,
                 number_of_warm_up_epochs=None,
                 log_directory=None,
                 **kwargs):

        # Class setup
        super().__init__()

        self.type = "GMVAE"

        self.feature_size = feature_size

        if latent_size is None:
            latent_size = defaults["models"]["latent_size"]
        self.latent_size = latent_size

        if hidden_sizes is None:
            hidden_sizes = defaults["models"]["hidden_sizes"]
        self.hidden_sizes = hidden_sizes

        if reconstruction_distribution is None:
            reconstruction_distribution = defaults["models"][
                "reconstruction_distribution"]
        reconstruction_distribution = parse_distribution(
            reconstruction_distribution)
        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = DISTRIBUTIONS[
            reconstruction_distribution]

        # Number of categorical elements needed for reconstruction, e.g. K+1
        if number_of_reconstruction_classes is None:
            number_of_reconstruction_classes = defaults["models"][
                "number_of_reconstruction_classes"]
        self.number_of_reconstruction_classes = (
            number_of_reconstruction_classes + 1)
        # K: For the sum over K-1 Categorical probabilities and the last K
        #   count distribution pdf.
        self.k_max = number_of_reconstruction_classes

        if latent_distribution is None:
            latent_distribution = defaults["models"]["latent_distribution"][
                self.type]
        latent_distribution = parse_distribution(
            latent_distribution, model_type=self.type)
        self.latent_distribution = copy.deepcopy(
            GAUSSIAN_MIXTURE_DISTRIBUTIONS[latent_distribution]
        )
        analytical_kl_term = False
        if latent_distribution == "legacy gaussian mixture":
            latent_distribution = "gaussian mixture"
            analytical_kl_term = True
        self.latent_distribution_name = latent_distribution
        self.analytical_kl_term = analytical_kl_term

        if number_of_latent_clusters is None:
            number_of_latent_clusters = defaults["models"]["number_of_classes"]
        self.n_clusters = number_of_latent_clusters

        if prior_probabilities_method is None:
            prior_probabilities_method = defaults["models"][
                "prior_probabilities_method"]
        if prior_probabilities_method in ["uniform", "learn"]:
            prior_probabilities = None
        elif prior_probabilities_method == "custom":
            if prior_probabilities is None:
                raise TypeError("No custom prior probabilities")
            elif isinstance(prior_probabilities, dict):
                prior_probabilities = list(prior_probabilities.values())
        else:
            raise NotImplementedError(
                "`{}` method for setting prior probabilities not implemented."
                .format(prior_probabilities_method)
            )
        self.prior_probabilities_method = prior_probabilities_method
        self.prior_probabilities = prior_probabilities

        if (self.prior_probabilities
                and len(self.prior_probabilities) != self.n_clusters):
            raise ValueError(
                "The number of provided prior probabilities has to be the "
                "same as the number of latent clusters."
            )

        # Dictionary holding number of samples needed for the Monte Carlo
        # estimator and importance weighting during both train and test time
        number_of_monte_carlo_samples = kwargs.get(
            "number_of_monte_carlo_samples")
        if number_of_monte_carlo_samples is None:
            number_of_monte_carlo_samples = defaults["models"][
                "number_of_samples"]
        else:
            number_of_monte_carlo_samples = parse_numbers_of_samples(
                number_of_monte_carlo_samples)
        self.number_of_monte_carlo_samples = number_of_monte_carlo_samples

        number_of_importance_samples = kwargs.get(
            "number_of_importance_samples")
        if number_of_importance_samples is None:
            number_of_importance_samples = defaults["models"][
                "number_of_samples"]
        else:
            number_of_importance_samples = parse_numbers_of_samples(
                number_of_importance_samples)
        self.number_of_importance_samples = number_of_importance_samples

        if minibatch_normalisation is None:
            minibatch_normalisation = defaults["models"][
                "minibatch_normalisation"]
        self.minibatch_normalisation = minibatch_normalisation

        if batch_correction is None:
            batch_correction = defaults["models"]["batch_correction"]
        self.batch_correction = batch_correction

        if self.batch_correction:
            if number_of_batches is None:
                raise TypeError(
                    "The number of batches for batch correction was not "
                    "provided."
                )
        self.number_of_batches = number_of_batches

        proportion_of_free_nats_for_y_kl_divergence = kwargs.get(
            "proportion_of_free_nats_for_y_kl_divergence")
        if proportion_of_free_nats_for_y_kl_divergence is None:
            proportion_of_free_nats_for_y_kl_divergence = defaults["models"][
                "proportion_of_free_nats_for_y_kl_divergence"]
        self.proportion_of_free_nats_for_y_kl_divergence = (
            proportion_of_free_nats_for_y_kl_divergence)

        # Dropout keep probabilities (p) for 3 different kinds of layers
        dropout_keep_probabilities = kwargs.get("dropout_keep_probabilities")
        if dropout_keep_probabilities is None:
            self.dropout_keep_probabilities = defaults["models"][
                "dropout_keep_probabilities"]
        else:
            self.dropout_keep_probabilities = dropout_keep_probabilities
        self.dropout_keep_probability_y = False
        self.dropout_keep_probability_z = False
        self.dropout_keep_probability_x = False
        self.dropout_keep_probability_h = False
        self.dropout_parts = []
        if isinstance(dropout_keep_probabilities, (list, tuple)):
            p_len = len(dropout_keep_probabilities)
            if p_len >= 4:
                self.dropout_keep_probability_y = dropout_keep_probabilities[3]
            if p_len >= 3:
                self.dropout_keep_probability_z = dropout_keep_probabilities[2]
            if p_len >= 2:
                self.dropout_keep_probability_x = dropout_keep_probabilities[1]
            if p_len >= 1:
                self.dropout_keep_probability_h = dropout_keep_probabilities[0]
            for i, p in enumerate(dropout_keep_probabilities):
                if p and p != 1:
                    self.dropout_parts.append(str(p))
        else:
            self.dropout_keep_probability_h = dropout_keep_probabilities
            if dropout_keep_probabilities and dropout_keep_probabilities != 1:
                self.dropout_parts.append(str(dropout_keep_probabilities))

        count_sum = kwargs.get("count_sum")
        if count_sum is None:
            count_sum = defaults["models"]["count_sum"]
        self.use_count_sum_as_feature = count_sum
        self.use_count_sum_as_parameter = (
            "constrained" in self.reconstruction_distribution_name
            or "multinomial" in self.reconstruction_distribution_name
        )

        kl_weight = kwargs.get("kl_weight")
        if kl_weight is None:
            kl_weight = defaults["models"]["kl_weight"]
        self.kl_weight_value = kl_weight

        if number_of_warm_up_epochs is None:
            number_of_warm_up_epochs = defaults["models"][
                "number_of_warm_up_epochs"]
        self.number_of_warm_up_epochs = number_of_warm_up_epochs

        if log_directory is None:
            log_directory = defaults["models"]["directory"]
        self.base_log_directory = log_directory

        # Early stopping
        self.early_stopping_rounds = 10
        self.stopped_early = None

        # Graph setup
        self.graph = tf.Graph()
        self.parameter_summary_list = []

        validate_model_parameters(
            reconstruction_distribution=self.reconstruction_distribution_name,
            number_of_reconstruction_classes=self.k_max
        )

        with self.graph.as_default():

            self.x = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.feature_size],
                name="X"
            )
            self.t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.feature_size],
                name="T"
            )

            self.learning_rate = tf.placeholder(
                dtype=tf.float32,
                shape=[],
                name="learning_rate"
            )

            self.kl_weight = tf.constant(
                self.kl_weight_value,
                dtype=tf.float32,
                name="kl_weight"
            )

            self.warm_up_weight = tf.placeholder(
                dtype=tf.float32,
                shape=[],
                name="warm_up_weight"
            )
            parameter_summary = tf.summary.scalar(
                name="warm_up_weight",
                tensor=self.warm_up_weight
            )
            self.parameter_summary_list.append(parameter_summary)

            self.total_kl_weight = tf.multiply(
                self.warm_up_weight,
                self.kl_weight,
                name="total_kl_weight"
            )
            total_kl_weight_summary = tf.summary.scalar(
                name="total_kl_weight",
                tensor=self.total_kl_weight
            )
            self.parameter_summary_list.append(total_kl_weight_summary)

            self.is_training = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name="is_training"
            )

            self.n_iw_samples = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="number_of_iw_samples"
            )
            self.n_mc_samples = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="number_of_mc_samples"
            )

            if self.batch_correction:
                self.batch_indices = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, 1],
                    name="batch_indices"
                )

            if self.use_count_sum_as_feature:
                self.count_sum_feature = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, 1],
                    name="count_sum_feature"
                )
            if self.use_count_sum_as_parameter:
                self.count_sum_parameter = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, 1],
                    name="count_sum"
                )
                self.replicated_count_sum_parameter = tf.tile(
                    self.count_sum_parameter,
                    multiples=[self.n_iw_samples * self.n_mc_samples, 1]
                )

            self.sample_size = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="sample_size"
            )

            self._setup_model_graph()
            self._setup_loss_function()
            self._setup_optimiser()

            self.saver = tf.train.Saver(max_to_keep=1)

    @property
    def name(self):
        """Short name for model used in filenames."""

        latent_parts = [normalise_string(self.latent_distribution_name)]

        if "mixture" in self.latent_distribution_name:
            latent_parts.append("c_{}".format(self.n_clusters))

        if self.prior_probabilities_method != "uniform":
            latent_parts.append("p_" + self.prior_probabilities_method)

        reconstruction_parts = [normalise_string(
            self.reconstruction_distribution_name)]

        if self.k_max:
            reconstruction_parts.append("k_{}".format(self.k_max))

        if self.use_count_sum_as_feature:
            reconstruction_parts.append("sum")

        reconstruction_parts.append("l_{}".format(self.latent_size))
        reconstruction_parts.append(
            "h_" + "_".join(map(str, self.hidden_sizes)))

        reconstruction_parts.append(
            "mc_{}".format(self.number_of_monte_carlo_samples["training"]))
        reconstruction_parts.append(
            "iw_{}".format(self.number_of_importance_samples["training"]))

        if self.analytical_kl_term:
            reconstruction_parts.append("kl")

        if self.minibatch_normalisation:
            reconstruction_parts.append("bn")

        if self.batch_correction:
            reconstruction_parts.append("bc")

        if len(self.dropout_parts) > 0:
            reconstruction_parts.append(
                "dropout_" + "_".join(self.dropout_parts))

        if self.kl_weight_value != 1:
            reconstruction_parts.append(
                "klw_{}".format(self.kl_weight_value))

        if self.number_of_warm_up_epochs:
            reconstruction_parts.append(
                "wu_{}".format(self.number_of_warm_up_epochs))

        if self.proportion_of_free_nats_for_y_kl_divergence:
            reconstruction_parts.append(
                "fn_{}".format(
                    self.proportion_of_free_nats_for_y_kl_divergence))

        latent_part = "-".join(latent_parts)
        reconstruction_part = "-".join(reconstruction_parts)

        model_name = os.path.join(self.type, latent_part, reconstruction_part)

        return model_name

    @property
    def description(self):
        """Description of model."""

        description_parts = ["Model setup:"]

        description_parts.append("type: {}".format(self.type))
        description_parts.append("feature size: {}".format(self.feature_size))
        description_parts.append("latent size: {}".format(self.latent_size))
        description_parts.append("hidden sizes: {}".format(", ".join(
            map(str, self.hidden_sizes))))

        description_parts.append(
            "latent distribution: " + self.latent_distribution_name)
        if "mixture" in self.latent_distribution_name:
            description_parts.append("latent clusters: {}".format(
                self.n_clusters))
            description_parts.append(
                "prior probabilities: " + self.prior_probabilities_method)

        description_parts.append(
            "reconstruction distribution: "
            + self.reconstruction_distribution_name
        )
        if self.k_max > 0:
            description_parts.append(
                "reconstruction classes: {} (including 0s)".format(self.k_max)
            )

        mc_train = self.number_of_monte_carlo_samples["training"]
        mc_eval = self.number_of_monte_carlo_samples["evaluation"]

        if mc_train > 1 or mc_eval > 1:
            mc = "Monte Carlo samples: {}".format(mc_train)
            if mc_eval != mc_train:
                mc += " (training), {} (evaluation)".format(mc_eval)
            description_parts.append(mc)

        iw_train = self.number_of_importance_samples["training"]
        iw_eval = self.number_of_importance_samples["evaluation"]

        if iw_train > 1 or iw_eval > 1:
            iw = "importance samples: {}".format(iw_train)
            if iw_eval != iw_train:
                iw += " (training), {} (evaluation)".format(iw_eval)
            description_parts.append(iw)

        if self.kl_weight_value != 1:
            description_parts.append(
                "KL weight: {}".format(self.kl_weight_value))

        if self.minibatch_normalisation:
            description_parts.append(
                "using batch normalisation for minibatches")

        if self.batch_correction:
            description_parts.append("with batch correction")

        if self.number_of_warm_up_epochs:
            description_parts.append(
                "using linear warm-up weighting for the first {} epochs"
                .format(self.number_of_warm_up_epochs)
            )

        if self.proportion_of_free_nats_for_y_kl_divergence:
            description_parts.append(
                "proportion of free nats for y KL divergence: {}"
                .format(self.proportion_of_free_nats_for_y_kl_divergence)
            )

        if len(self.dropout_parts) > 0:
            description_parts.append(
                "dropout keep probability: {}".format(
                    ", ".join(self.dropout_parts)
                )
            )

        if self.use_count_sum_as_feature:
            description_parts.append("using count sums")

        if self.early_stopping_rounds:
            description_parts.append(
                "early stopping: after {} epoch with no improvements"
                .format(self.early_stopping_rounds)
            )

        description = "\n    ".join(description_parts)

        return description

    @property
    def parameters(self):
        """Trainable parameters in the model."""

        with self.graph.as_default():
            parameters = tf.trainable_variables()

        parameters_string_parts = ["Trainable parameters"]
        width = max(map(len, [p.name for p in parameters]))

        for parameter in parameters:
            parameters_string_parts.append(
                "{:{}}  {}".format(
                    parameter.name, width, parameter.get_shape()
                )
            )

        parameters_string = "\n    ".join(parameters_string_parts)

        return parameters_string

    @property
    def number_of_latent_clusters(self):
        """The number of latent clusters used in the model."""
        return self.n_clusters

    def has_been_trained(self, run_id=None):
        log_directory = self.log_directory(run_id=run_id)
        checkpoint = tf.train.get_checkpoint_state(log_directory)
        return bool(checkpoint)

    def log_directory(self, base=None, run_id=None,
                      early_stopping=False, best_model=False):

        if not base:
            base = self.base_log_directory

        log_directory = os.path.join(base, self.name)

        if run_id is None:
            run_id = defaults["models"]["run_id"]
        if run_id:
            run_id = check_run_id(run_id)
            log_directory = os.path.join(
                log_directory,
                "run_{}".format(run_id)
            )

        if early_stopping and best_model:
            raise ValueError(
                "Early-stopping model and best model are mutually exclusive."
            )
        elif early_stopping:
            log_directory = os.path.join(log_directory, "early_stopping")
        elif best_model:
            log_directory = os.path.join(log_directory, "best")

        return log_directory

    def early_stopping_status(self, run_id=None):

        stopped_early = False
        epochs_with_no_improvement = 0

        early_stopping_log_directory = self.log_directory(
            run_id=run_id,
            early_stopping=True
        )

        log_directory = os.path.dirname(early_stopping_log_directory)

        if os.path.exists(log_directory):

            validation_losses = load_learning_curves(
                model=self,
                data_set_kinds="validation",
                run_id=run_id,
                log_directory=log_directory
            )["lower_bound"]

            if os.path.exists(early_stopping_log_directory):
                stopped_early, epochs_with_no_improvement = (
                    early_stopping_status(
                        validation_losses,
                        self.early_stopping_rounds
                    )
                )

        return stopped_early, epochs_with_no_improvement

    def train(self, training_set, validation_set=None, number_of_epochs=None,
              minibatch_size=None, learning_rate=None, run_id=None,
              new_run=False, reset_training=False, **kwargs):
        """Train model.

        Arguments:
            training_set (DataSet): Data set used to train model.
            validation_set (DataSet, optional): Data set used to
                validate model during training, if given.
            number_of_epochs (int, optional): The number of epochs to
                train the model.
            minibatch_size (int, optional): The size of the random
                minibatches used at each step of training.
            learning_rate (float, optional): The learning rate used at
                each step of training.
            run_id (str, optional): ID used to identify a certain run
                of the model.
            new_run (bool, optional): If ``True``, train a model anew
                as a separate run with an automatically generated ID.
            reset_training (bool, optional): If ``True``, reset model
                by removing saved parameters for the model.
        """

        if number_of_epochs is None:
            number_of_epochs = defaults["models"]["number_of_epochs"]
        if minibatch_size is None:
            minibatch_size = defaults["models"]["minibatch_size"]
        if learning_rate is None:
            learning_rate = defaults["models"]["learning_rate"]
        if run_id is None:
            run_id = defaults["models"]["run_id"]
        if new_run is None:
            new_run = defaults["models"]["new_run"]
        if reset_training is None:
            reset_training = defaults["models"]["reset_training"]

        analyses_directory = kwargs.get("analyses_directory")
        if analyses_directory is None:
            analyses_directory = defaults["analyses"]["directory"]

        start_time = time()

        if run_id is None:
            run_id = defaults["models"]["run_id"]
        if run_id:
            run_id = check_run_id(run_id)
            new_run = True
        elif new_run:
            run_id = generate_unique_run_id_for_model(
                model=self,
                timestamp=start_time
            )

        if run_id:
            model_string = "model for run {}".format(run_id)
        else:
            model_string = "model"

        # Remove model run if prompted
        permanent_log_directory = self.log_directory(run_id=run_id)
        if reset_training and os.path.exists(permanent_log_directory):
            clear_log_directory(permanent_log_directory)

        # Logging
        metadata_log = {
            "epochs trained": None,
            "start time": format_time(start_time),
            "training duration": None,
            "last epoch duration": None,
            "learning rate": learning_rate,
            "minibatch size": minibatch_size
        }

        # Earlier model
        old_checkpoint = tf.train.get_checkpoint_state(permanent_log_directory)
        if old_checkpoint:
            epoch_start = int(os.path.basename(
                old_checkpoint.model_checkpoint_path).split("-")[-1])
        else:
            epoch_start = 0

        # Log directories
        temporary_log_directory = kwargs.get("temporary_log_directory")
        if temporary_log_directory:

            log_directory = self.log_directory(
                base=temporary_log_directory,
                run_id=run_id
            )
            early_stopping_log_directory = self.log_directory(
                base=temporary_log_directory,
                run_id=run_id,
                early_stopping=True
            )
            best_model_log_directory = self.log_directory(
                base=temporary_log_directory,
                run_id=run_id,
                best_model=True
            )

            temporary_checkpoint = (
                tf.train.get_checkpoint_state(log_directory))

            if temporary_checkpoint:
                temporary_epoch_start = int(os.path.basename(
                    temporary_checkpoint.model_checkpoint_path
                ).split("-")[-1])
            else:
                temporary_epoch_start = 0

            if temporary_epoch_start > epoch_start:
                epoch_start = temporary_epoch_start
                replace_temporary_directory = False
            else:
                replace_temporary_directory = True

        else:
            log_directory = self.log_directory(run_id=run_id)
            early_stopping_log_directory = self.log_directory(
                run_id=run_id,
                early_stopping=True
            )
            best_model_log_directory = self.log_directory(
                run_id=run_id,
                best_model=True
            )

        # Training message
        data_string = build_data_string(
            data_set=training_set,
            reconstruction_distribution_name=(
                self.reconstruction_distribution_name)
        )
        training_string = build_training_string(
            model_string,
            epoch_start=epoch_start,
            number_of_epochs=number_of_epochs,
            data_string=data_string
        )

        # Stop, if model is already trained
        if epoch_start >= number_of_epochs:
            print(training_string)
            return 0

        # Copy log directory to temporary location, if necessary
        if (temporary_log_directory
                and os.path.exists(permanent_log_directory)
                and replace_temporary_directory):

            print("Copying log directory to temporary directory.")
            copying_time_start = time()

            if os.path.exists(log_directory):
                shutil.rmtree(log_directory)

            shutil.copytree(permanent_log_directory, log_directory)

            copying_duration = time() - copying_time_start
            print("Log directory copied ({}).".format(format_duration(
                copying_duration)))

            print()

        # New model
        checkpoint_file = os.path.join(log_directory, "model.ckpt")

        print("Preparing data.")
        preparing_data_time_start = time()

        # Batch indices for batch correction
        if self.batch_correction:
            batch_indices_train = batch_indices_for_subset(training_set)
            if validation_set:
                batch_indices_valid = batch_indices_for_subset(validation_set)

        # Count sum for distributions
        if self.use_count_sum_as_parameter:
            count_sum_parameter_train = training_set.count_sum
            if validation_set:
                count_sum_parameter_valid = validation_set.count_sum

        # Normalised count sum as a feature to the decoder
        if self.use_count_sum_as_feature:
            count_sum_feature_train = training_set.normalised_count_sum
            if validation_set:
                count_sum_feature_valid = validation_set.normalised_count_sum

        # Numbers of examples for data subsets
        n_examples_train = training_set.number_of_examples
        if validation_set:
            n_examples_valid = validation_set.number_of_examples

        # Preprocessing function at every epoch
        noisy_preprocess = training_set.noisy_preprocess

        # Input and output
        if not noisy_preprocess:

            if training_set.has_preprocessed_values:
                x_train = training_set.preprocessed_values
                if validation_set:
                    x_valid = validation_set.preprocessed_values
            else:
                x_train = training_set.values
                if validation_set:
                    x_valid = validation_set.values

            if self.reconstruction_distribution_name == "bernoulli":
                t_train = training_set.binarised_values
                if validation_set:
                    t_valid = validation_set.binarised_values
            else:
                t_train = training_set.values
                if validation_set:
                    t_valid = validation_set.values

        # Use label IDs instead of labels
        if training_set.has_labels:

            class_names_to_class_ids = numpy.vectorize(
                lambda class_name:
                    training_set.class_name_to_class_id[class_name]
            )

            training_label_ids = class_names_to_class_ids(training_set.labels)
            if validation_set:
                validation_label_ids = class_names_to_class_ids(
                    validation_set.labels)

            if training_set.excluded_classes:
                excluded_class_ids = class_names_to_class_ids(
                    training_set.excluded_classes)
            else:
                excluded_class_ids = []

        # Use superset label IDs instead of superset labels
        if training_set.has_superset_labels:

            superset_class_names_to_superset_class_ids = numpy.vectorize(
                lambda superset_class_name:
                    training_set.superset_class_name_to_superset_class_id[
                        superset_class_name]
            )

            training_superset_label_ids = (
                superset_class_names_to_superset_class_ids(
                    training_set.superset_labels))
            if validation_set:
                validation_superset_label_ids = (
                    superset_class_names_to_superset_class_ids(
                        validation_set.superset_labels))

            if training_set.excluded_superset_classes:
                excluded_superset_class_ids = (
                    superset_class_names_to_superset_class_ids(
                        training_set.excluded_superset_classes))
            else:
                excluded_superset_class_ids = []

        preparing_data_duration = time() - preparing_data_time_start
        print("Data prepared ({}).".format(format_duration(
            preparing_data_duration)))
        print()

        # Display intervals during every epoch
        steps_per_epoch = numpy.ceil(n_examples_train / minibatch_size)
        output_at_step = numpy.round(numpy.linspace(0, steps_per_epoch, 11))

        # Initialising lists for learning curves
        learning_curves = {
            "training": {
                "lower_bound": [],
                "reconstruction_error": [],
                "kl_divergence_z": [],
                "kl_divergence_y": []
            }
        }
        if validation_set:
            learning_curves["validation"] = {
                "lower_bound": [],
                "reconstruction_error": [],
                "kl_divergence_z": [],
                "kl_divergence_y": []
            }

        with tf.Session(graph=self.graph) as session:

            parameter_summary_writer = tf.summary.FileWriter(log_directory)
            training_summary_writer = tf.summary.FileWriter(
                os.path.join(log_directory, "training"))
            if validation_set:
                validation_summary_writer = tf.summary.FileWriter(
                    os.path.join(log_directory, "validation"))

            # Initialisation

            checkpoint = tf.train.get_checkpoint_state(log_directory)

            if checkpoint:
                print("Restoring earlier model parameters.")
                restoring_time_start = time()

                model_checkpoint_path = correct_model_checkpoint_path(
                    checkpoint.model_checkpoint_path,
                    log_directory
                )
                self.saver.restore(session, model_checkpoint_path)
                epoch_start = int(
                    os.path.split(model_checkpoint_path)[-1].split("-")[-1])

                if validation_set:
                    lower_bound_valid_learning_curve = load_learning_curves(
                        model=self,
                        data_set_kinds="validation",
                        run_id=run_id,
                        log_directory=log_directory
                    )["lower_bound"]

                    lower_bound_valid_maximum = (
                        lower_bound_valid_learning_curve.max())

                    self.stopped_early, epochs_with_no_improvement = (
                        self.early_stopping_status(run_id=run_id))
                    lower_bound_valid_early_stopping = (
                        lower_bound_valid_learning_curve[
                            -1 - epochs_with_no_improvement])

                restoring_duration = time() - restoring_time_start
                print("Earlier model parameters restored ({}).".format(
                    format_duration(restoring_duration)))
                print()
            else:
                print("Initialising model parameters.")
                initialising_time_start = time()

                session.run(tf.global_variables_initializer())
                parameter_summary_writer.add_graph(session.graph)
                epoch_start = 0

                if validation_set:
                    lower_bound_valid_maximum = - numpy.inf
                    epochs_with_no_improvement = 0
                    lower_bound_valid_early_stopping = - numpy.inf
                    self.stopped_early = False

                initialising_duration = time() - initialising_time_start
                print("Model parameters initialised ({}).".format(
                    format_duration(initialising_duration)))
                print()

            metadata_log["epochs trained"] = (epoch_start, number_of_epochs)

            print(training_string)
            print()
            training_time_start = time()

            for epoch in range(epoch_start, number_of_epochs):

                if noisy_preprocess:
                    print("Noisily preprocess values.")
                    noisy_time_start = time()

                    x_train = noisy_preprocess(
                        training_set.values)
                    t_train = x_train

                    if validation_set:
                        x_valid = noisy_preprocess(
                            validation_set.values)
                        t_valid = x_valid

                    noisy_duration = time() - noisy_time_start
                    print("Values noisily preprocessed ({}).".format(
                        format_duration(noisy_duration)))
                    print()

                epoch_time_start = time()

                if self.number_of_warm_up_epochs:
                    warm_up_weight = float(
                        min(epoch / (self.number_of_warm_up_epochs), 1.0))
                else:
                    warm_up_weight = 1.0

                shuffled_indices = numpy.random.permutation(n_examples_train)

                for i in range(0, n_examples_train, minibatch_size):

                    # Internal setup
                    step_time_start = time()
                    step = session.run(self.global_step)

                    # Prepare minibatch
                    minibatch_indices = shuffled_indices[
                        i:(i + minibatch_size)]

                    x_batch = x_train[minibatch_indices].toarray()
                    t_batch = t_train[minibatch_indices].toarray()

                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: True,
                        self.learning_rate: learning_rate,
                        self.warm_up_weight: warm_up_weight,
                        self.n_iw_samples:
                            self.number_of_importance_samples["training"],
                        self.n_mc_samples:
                            self.number_of_monte_carlo_samples["training"]
                    }

                    if self.batch_correction:
                        feed_dict_batch[self.batch_indices] = (
                            batch_indices_train[minibatch_indices])

                    if self.use_count_sum_as_parameter:
                        feed_dict_batch[self.count_sum_parameter] = (
                            count_sum_parameter_train[minibatch_indices])

                    if self.use_count_sum_as_feature:
                        feed_dict_batch[self.count_sum_feature] = (
                            count_sum_feature_train[minibatch_indices])

                    # Run the stochastic minibatch training operation
                    _, minibatch_loss = session.run(
                        [self.optimiser, self.lower_bound],
                        feed_dict=feed_dict_batch
                    )

                    # Compute step duration
                    step_duration = time() - step_time_start

                    # Print evaluation and output summaries
                    if (step + 1 - steps_per_epoch * epoch) in output_at_step:

                        print("Step {:d} ({}): {:.5g}.".format(
                            int(step + 1), format_duration(step_duration),
                            minibatch_loss))

                        if numpy.isnan(minibatch_loss):
                            raise ArithmeticError("The loss became NaN.")

                print()

                epoch_duration = time() - epoch_time_start

                print("Epoch {} ({}):".format(
                    epoch + 1, format_duration(epoch_duration)))

                # With warmup or not
                if warm_up_weight < 1:
                    print("    Warm-up weight: {:.2g}".format(warm_up_weight))

                # Export parameter summaries
                parameter_summary_string = session.run(
                    self.parameter_summary,
                    feed_dict={self.warm_up_weight: warm_up_weight}
                )
                parameter_summary_writer.add_summary(
                    parameter_summary_string, global_step=epoch + 1)
                parameter_summary_writer.flush()

                print("    Evaluating model.")

                # Training evaluation
                evaluating_time_start = time()

                lower_bound_train = 0
                kl_divergence_z_train = 0
                kl_divergence_y_train = 0
                reconstruction_error_train = 0

                q_y_probabilities = numpy.zeros(shape=self.n_clusters)
                q_z_means = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))
                q_z_variances = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))

                p_y_probabilities = numpy.zeros(shape=self.n_clusters)
                p_z_means = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))
                p_z_variances = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))

                if "full-covariance" in self.latent_distribution_name:
                    q_z_covariances = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size,
                               self.latent_size))
                    p_z_covariances = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size,
                               self.latent_size))

                q_y_logits_train = numpy.zeros(
                    shape=(n_examples_train, self.n_clusters),
                    dtype=numpy.float32
                )
                z_mean_train = numpy.zeros(
                    shape=(n_examples_train, self.latent_size),
                    dtype=numpy.float32
                )

                if "mixture" in self.latent_distribution_name:
                    kl_divergence_neurons = numpy.zeros(1)
                else:
                    kl_divergence_neurons = numpy.zeros(self.latent_size)

                for i in range(0, n_examples_train, minibatch_size):
                    subset = slice(
                        i, min(i + minibatch_size, n_examples_train))
                    x_batch = x_train[subset].toarray()
                    t_batch = t_train[subset].toarray()
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.warm_up_weight: 1.0,
                        self.n_iw_samples:
                            self.number_of_importance_samples["training"],
                        self.n_mc_samples:
                            self.number_of_monte_carlo_samples["training"]
                    }

                    if self.batch_correction:
                        feed_dict_batch[self.batch_indices] = (
                            batch_indices_train[subset])

                    if self.use_count_sum_as_parameter:
                        feed_dict_batch[self.count_sum_parameter] = (
                            count_sum_parameter_train[subset])

                    if self.use_count_sum_as_feature:
                        feed_dict_batch[self.count_sum_feature] = (
                            count_sum_feature_train[subset])

                    (
                        lower_bound_i, reconstruction_error_i,
                        kl_divergence_z_i, kl_divergence_y_i,
                        kl_divergence_neurons_i,
                        q_y_probabilities_i, q_z_means_i, q_z_variances_i,
                        p_y_probabilities_i, p_z_means_i, p_z_variances_i,
                        q_z_covariances_i, p_z_covariances_i,
                        q_y_logits_train_i, z_mean_i
                    ) = session.run(
                        [
                            self.lower_bound, self.reconstruction_error,
                            self.kl_divergence_z, self.kl_divergence_y,
                            self.kl_divergence_neurons, self.q_y_probabilities,
                            self.q_z_means, self.q_z_variances,
                            self.p_y_probabilities, self.p_z_means,
                            self.p_z_variances, self.q_z_covariances,
                            self.p_z_covariances, self.q_y_logits, self.z_mean
                        ],
                        feed_dict=feed_dict_batch
                    )

                    lower_bound_train += lower_bound_i
                    kl_divergence_z_train += kl_divergence_z_i
                    kl_divergence_y_train += kl_divergence_y_i
                    reconstruction_error_train += reconstruction_error_i

                    kl_divergence_neurons += kl_divergence_neurons_i

                    q_y_probabilities += numpy.array(q_y_probabilities_i)
                    q_z_means += numpy.array(q_z_means_i)
                    q_z_variances += numpy.array(q_z_variances_i)

                    p_y_probabilities += numpy.array(p_y_probabilities_i)
                    p_z_means += numpy.array(p_z_means_i)
                    p_z_variances += numpy.array(p_z_variances_i)

                    if "full-covariance" in self.latent_distribution_name:
                        q_z_covariances += numpy.array(q_z_covariances_i)
                        p_z_covariances += numpy.array(p_z_covariances_i)

                    q_y_logits_train[subset] = q_y_logits_train_i
                    z_mean_train[subset] = z_mean_i

                lower_bound_train /= n_examples_train / minibatch_size
                kl_divergence_z_train /= n_examples_train / minibatch_size
                kl_divergence_y_train /= n_examples_train / minibatch_size
                reconstruction_error_train /= n_examples_train / minibatch_size

                kl_divergence_neurons /= n_examples_train / minibatch_size

                q_y_probabilities /= n_examples_train / minibatch_size
                q_z_means /= n_examples_train / minibatch_size
                q_z_variances /= n_examples_train / minibatch_size

                p_y_probabilities /= n_examples_train / minibatch_size
                p_z_means /= n_examples_train / minibatch_size
                p_z_variances /= n_examples_train / minibatch_size

                if "full-covariance" in self.latent_distribution_name:
                    q_z_covariances /= n_examples_train / minibatch_size
                    p_z_covariances /= n_examples_train / minibatch_size

                learning_curves["training"]["lower_bound"].append(
                    lower_bound_train)
                learning_curves["training"]["reconstruction_error"].append(
                    reconstruction_error_train)
                learning_curves["training"]["kl_divergence_z"].append(
                    kl_divergence_z_train)
                learning_curves["training"]["kl_divergence_y"].append(
                    kl_divergence_y_train)

                # Training accuracies

                training_cluster_ids = q_y_logits_train.argmax(axis=1)

                if training_set.has_labels:
                    predicted_training_label_ids = (
                        map_cluster_ids_to_label_ids(
                            training_label_ids,
                            training_cluster_ids,
                            excluded_class_ids
                        )
                    )
                    accuracy_train = accuracy(
                        training_label_ids,
                        predicted_training_label_ids,
                        excluded_class_ids
                    )
                else:
                    accuracy_train = None

                if training_set.label_superset:
                    predicted_training_superset_label_ids = (
                        map_cluster_ids_to_label_ids(
                            training_superset_label_ids,
                            training_cluster_ids,
                            excluded_superset_class_ids
                        )
                    )
                    accuracy_superset_train = accuracy(
                        training_superset_label_ids,
                        predicted_training_superset_label_ids,
                        excluded_superset_class_ids
                    )
                    accuracy_display = accuracy_superset_train
                else:
                    accuracy_superset_train = None
                    accuracy_display = accuracy_train

                evaluating_duration = time() - evaluating_time_start

                # Training summaries
                summary = tf.Summary()

                # Training loss and accuracy summaries
                summary.value.add(
                    tag="losses/lower_bound",
                    simple_value=lower_bound_train
                )
                summary.value.add(
                    tag="losses/reconstruction_error",
                    simple_value=reconstruction_error_train
                )
                summary.value.add(
                    tag="losses/kl_divergence_z",
                    simple_value=kl_divergence_z_train
                )
                summary.value.add(
                    tag="losses/kl_divergence_y",
                    simple_value=kl_divergence_y_train
                )
                summary.value.add(
                    tag="accuracy",
                    simple_value=accuracy_train
                )
                if accuracy_superset_train:
                    summary.value.add(
                        tag="superset_accuracy",
                        simple_value=accuracy_superset_train
                    )

                # Training KL divergence summaries
                for i in range(kl_divergence_neurons.size):
                    summary.value.add(
                        tag="kl_divergence_neurons/{}".format(i),
                        simple_value=kl_divergence_neurons[i]
                    )

                # Training centroid summaries
                if validation_set is None:
                    for k in range(self.n_clusters):
                        summary.value.add(
                            tag="prior/cluster_{}/probability".format(k),
                            simple_value=p_y_probabilities[k]
                        )
                        summary.value.add(
                            tag="posterior/cluster_{}/probability".format(k),
                            simple_value=q_y_probabilities[k]
                        )
                        for l in range(self.latent_size):
                            summary.value.add(
                                tag="prior/cluster_{}/mean/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_means[k][l]
                            )
                            summary.value.add(
                                tag="posterior/cluster_{}/mean/dimension_{}"
                                    .format(k, l),
                                simple_value=q_z_means[k, l]
                            )
                            summary.value.add(
                                tag="prior/cluster_{}/variance/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_variances[k][l]
                            )
                            summary.value.add(
                                tag="posterior/cluster_{}/variance/"
                                    "dimension_{}".format(k, l),
                                simple_value=q_z_variances[k, l]
                            )
                            if "full-covariance" in (
                                    self.latent_distribution_name):
                                for l_ in range(self.latent_size):
                                    summary.value.add(
                                        tag="prior/cluster_{}/covariance/"
                                            "dimension_{}_{}".format(k, l, l_),
                                        simple_value=p_z_covariances[k, l, l_]
                                    )
                                    summary.value.add(
                                        tag="posterior/cluster_{}/covariance/"
                                            "dimension_{}_{}".format(k, l, l_),
                                        simple_value=q_z_covariances[k, l, l_]
                                    )

                # Writing training summaries
                training_summary_writer.add_summary(
                    summary, global_step=epoch + 1)
                training_summary_writer.flush()

                # Printing training evaluation
                evaluation_string = "    {} set ({}): ".format(
                    training_set.kind.capitalize(),
                    format_duration(evaluating_duration)
                )
                evaluation_metrics = [
                    "ELBO: {:.5g}".format(lower_bound_train),
                    "ENRE: {:.5g}".format(reconstruction_error_train),
                    "KL_z: {:.5g}".format(kl_divergence_z_train),
                    "KL_y: {:.5g}".format(kl_divergence_y_train)
                ]
                if accuracy_display:
                    evaluation_metrics.append(
                        "Acc: {:.5g}".format(accuracy_display)
                    )
                evaluation_string += ", ".join(evaluation_metrics)
                evaluation_string += "."

                print(evaluation_string)

                if validation_set:

                    # Validation evaluation
                    evaluating_time_start = time()

                    lower_bound_valid = 0
                    kl_divergence_z_valid = 0
                    kl_divergence_y_valid = 0
                    reconstruction_error_valid = 0

                    q_y_probabilities = numpy.zeros(shape=self.n_clusters)
                    q_z_means = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size))
                    q_z_variances = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size))

                    p_y_probabilities = numpy.zeros(shape=self.n_clusters)
                    p_z_means = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size))
                    p_z_variances = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size))

                    if "full-covariance" in self.latent_distribution_name:
                        q_z_covariances = numpy.zeros(
                            shape=(self.n_clusters, self.latent_size,
                                   self.latent_size))
                        p_z_covariances = numpy.zeros(
                            shape=(self.n_clusters, self.latent_size,
                                   self.latent_size))

                    q_y_logits_valid = numpy.zeros(
                        shape=(n_examples_valid, self.n_clusters),
                        dtype=numpy.float32
                    )
                    z_mean_valid = numpy.zeros(
                        shape=(n_examples_valid, self.latent_size),
                        dtype=numpy.float32
                    )

                    for i in range(0, n_examples_valid, minibatch_size):
                        subset = slice(
                            i, min(i + minibatch_size, n_examples_valid))
                        x_batch = x_valid[subset].toarray()
                        t_batch = t_valid[subset].toarray()
                        feed_dict_batch = {
                            self.x: x_batch,
                            self.t: t_batch,
                            self.is_training: False,
                            self.warm_up_weight: 1.0,
                            self.n_iw_samples:
                                self.number_of_importance_samples["training"],
                            self.n_mc_samples:
                                self.number_of_monte_carlo_samples["training"]
                        }

                        if self.batch_correction:
                            feed_dict_batch[self.batch_indices] = (
                                batch_indices_valid[subset])

                        if self.use_count_sum_as_parameter:
                            feed_dict_batch[self.count_sum_parameter] = (
                                count_sum_parameter_valid[subset])

                        if self.use_count_sum_as_feature:
                            feed_dict_batch[self.count_sum_feature] = (
                                count_sum_feature_valid[subset])

                        (
                            lower_bound_i, reconstruction_error_i,
                            kl_divergence_z_i, kl_divergence_y_i,
                            q_y_probabilities_i, q_z_means_i, q_z_variances_i,
                            p_y_probabilities_i, p_z_means_i, p_z_variances_i,
                            q_z_covariances_i, p_z_covariances_i,
                            q_y_logits_i, z_mean_i
                        ) = session.run(
                            [
                                self.lower_bound, self.reconstruction_error,
                                self.kl_divergence_z, self.kl_divergence_y,
                                self.q_y_probabilities, self.q_z_means,
                                self.q_z_variances, self.p_y_probabilities,
                                self.p_z_means, self.p_z_variances,
                                self.q_z_covariances, self.p_z_covariances,
                                self.q_y_logits, self.z_mean
                            ],
                            feed_dict=feed_dict_batch
                        )

                        lower_bound_valid += lower_bound_i
                        kl_divergence_z_valid += kl_divergence_z_i
                        kl_divergence_y_valid += kl_divergence_y_i
                        reconstruction_error_valid += reconstruction_error_i

                        q_y_probabilities += numpy.array(q_y_probabilities_i)
                        q_z_means += numpy.array(q_z_means_i)
                        q_z_variances += numpy.array(q_z_variances_i)

                        p_y_probabilities += numpy.array(p_y_probabilities_i)
                        p_z_means += numpy.array(p_z_means_i)
                        p_z_variances += numpy.array(p_z_variances_i)

                        if "full-covariance" in self.latent_distribution_name:
                            q_z_covariances += numpy.array(q_z_covariances_i)
                            p_z_covariances += numpy.array(p_z_covariances_i)

                        q_y_logits_valid[subset] = q_y_logits_i
                        z_mean_valid[subset] = z_mean_i

                    lower_bound_valid /= n_examples_valid / minibatch_size
                    kl_divergence_z_valid /= n_examples_valid / minibatch_size
                    kl_divergence_y_valid /= n_examples_valid / minibatch_size
                    reconstruction_error_valid /= (
                        n_examples_valid / minibatch_size)

                    q_y_probabilities /= n_examples_valid / minibatch_size
                    q_z_means /= n_examples_valid / minibatch_size
                    q_z_variances /= n_examples_valid / minibatch_size

                    p_y_probabilities /= n_examples_valid / minibatch_size
                    p_z_means /= n_examples_valid / minibatch_size
                    p_z_variances /= n_examples_valid / minibatch_size

                    if "full-covariance" in self.latent_distribution_name:
                        q_z_covariances /= n_examples_valid / minibatch_size
                        p_z_covariances /= n_examples_valid / minibatch_size

                    learning_curves["validation"]["lower_bound"].append(
                        lower_bound_valid)
                    learning_curves["validation"][
                        "reconstruction_error"].append(
                            reconstruction_error_valid)
                    learning_curves["validation"]["kl_divergence_z"].append(
                        kl_divergence_z_valid)
                    learning_curves["validation"]["kl_divergence_y"].append(
                        kl_divergence_y_valid)

                    # Validation accuracies
                    validation_cluster_ids = q_y_logits_valid.argmax(axis=1)

                    if validation_set.has_labels:
                        predicted_validation_label_ids = (
                            map_cluster_ids_to_label_ids(
                                validation_label_ids,
                                validation_cluster_ids,
                                excluded_class_ids
                            )
                        )
                        accuracy_valid = accuracy(
                            validation_label_ids,
                            predicted_validation_label_ids,
                            excluded_class_ids
                        )
                    else:
                        accuracy_valid = None

                    if validation_set.label_superset:
                        predicted_validation_superset_label_ids = (
                            map_cluster_ids_to_label_ids(
                                validation_superset_label_ids,
                                validation_cluster_ids,
                                excluded_superset_class_ids
                            )
                        )
                        accuracy_superset_valid = accuracy(
                            validation_superset_label_ids,
                            predicted_validation_superset_label_ids,
                            excluded_superset_class_ids
                        )
                        accuracy_display = accuracy_superset_valid
                    else:
                        accuracy_superset_valid = None
                        accuracy_display = accuracy_valid

                    evaluating_duration = time() - evaluating_time_start

                    # Validation summaries
                    summary = tf.Summary()

                    # Validation loss and accuracy summaries
                    summary.value.add(
                        tag="losses/lower_bound",
                        simple_value=lower_bound_valid
                    )
                    summary.value.add(
                        tag="losses/reconstruction_error",
                        simple_value=reconstruction_error_valid
                    )
                    summary.value.add(
                        tag="losses/kl_divergence_z",
                        simple_value=kl_divergence_z_valid
                    )
                    summary.value.add(
                        tag="losses/kl_divergence_y",
                        simple_value=kl_divergence_y_valid
                    )
                    summary.value.add(
                        tag="accuracy",
                        simple_value=accuracy_valid
                    )
                    if accuracy_superset_valid:
                        summary.value.add(
                            tag="superset_accuracy",
                            simple_value=accuracy_superset_valid
                        )

                    # Validation centroid summaries
                    for k in range(self.n_clusters):
                        summary.value.add(
                            tag="prior/cluster_{}/probability".format(k),
                            simple_value=p_y_probabilities[k]
                        )
                        summary.value.add(
                            tag="posterior/cluster_{}/probability".format(k),
                            simple_value=q_y_probabilities[k]
                        )
                        for l in range(self.latent_size):
                            summary.value.add(
                                tag="prior/cluster_{}/mean/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_means[k][l]
                            )
                            summary.value.add(
                                tag="posterior/cluster_{}/mean/dimension_{}"
                                    .format(k, l),
                                simple_value=q_z_means[k, l]
                            )
                            summary.value.add(
                                tag="prior/cluster_{}/variance/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_variances[k][l]
                            )
                            summary.value.add(
                                tag="posterior/cluster_{}/variance/"
                                    "dimension_{}".format(k, l),
                                simple_value=q_z_variances[k, l]
                            )
                            if "full-covariance" in (
                                    self.latent_distribution_name):
                                for l_ in range(self.latent_size):
                                    summary.value.add(
                                        tag="prior/cluster_{}/covariance/"
                                            "dimension_{}_{}".format(k, l, l_),
                                        simple_value=p_z_covariances[k, l, l_]
                                    )
                                    summary.value.add(
                                        tag="posterior/cluster_{}/covariance/"
                                            "dimension_{}_{}".format(k, l, l_),
                                        simple_value=q_z_covariances[k, l, l_]
                                    )

                    # Writing validation summaries
                    validation_summary_writer.add_summary(
                        summary, global_step=epoch + 1)
                    validation_summary_writer.flush()

                    # Printing validation evaluation
                    evaluation_string = "    {} set ({}): ".format(
                        validation_set.kind.capitalize(),
                        format_duration(evaluating_duration)
                    )
                    evaluation_metrics = [
                        "ELBO: {:.5g}".format(lower_bound_valid),
                        "ENRE: {:.5g}".format(reconstruction_error_valid),
                        "KL_z: {:.5g}".format(kl_divergence_z_valid),
                        "KL_y: {:.5g}".format(kl_divergence_y_valid)
                    ]
                    if accuracy_display:
                        evaluation_metrics.append(
                            "Acc: {:.5g}".format(accuracy_display)
                        )
                    evaluation_string += ", ".join(evaluation_metrics)
                    evaluation_string += "."

                    print(evaluation_string)

                # Early stopping
                if validation_set and not self.stopped_early:

                    if lower_bound_valid < lower_bound_valid_early_stopping:
                        if epochs_with_no_improvement == 0:
                            print(
                                "    Early stopping:",
                                "Validation loss did not improve",
                                "for this epoch."
                            )
                            print(
                                "        "
                                "Saving model parameters for previous epoch."
                            )
                            saving_time_start = time()
                            lower_bound_valid_early_stopping = (
                                lower_bound_valid)
                            current_checkpoint = tf.train.get_checkpoint_state(
                                log_directory)
                            if current_checkpoint:
                                copy_model_directory(
                                    current_checkpoint,
                                    early_stopping_log_directory
                                )
                            saving_duration = time() - saving_time_start
                            print(
                                "        "
                                "Previous model parameters saved ({})."
                                .format(format_duration(saving_duration))
                            )
                        else:
                            print(
                                "    Early stopping:",
                                "Validation loss has not improved",
                                "for {} epochs.".format(
                                    epochs_with_no_improvement + 1)
                            )
                        epochs_with_no_improvement += 1
                    else:
                        if epochs_with_no_improvement > 0:
                            print(
                                "    Early stopping cancelled:",
                                "Validation loss improved."
                            )
                        epochs_with_no_improvement = 0
                        lower_bound_valid_early_stopping = lower_bound_valid
                        if os.path.exists(early_stopping_log_directory):
                            shutil.rmtree(early_stopping_log_directory)

                    if (epochs_with_no_improvement
                            >= self.early_stopping_rounds):

                        print(
                            "    Early stopping in effect:",
                            "Previously saved model parameters is available."
                        )
                        self.stopped_early = True
                        epochs_with_no_improvement = numpy.nan

                # Saving model parameters (update checkpoint)
                print("    Saving model parameters.")
                saving_time_start = time()
                self.saver.save(
                    session, checkpoint_file, global_step=epoch + 1)
                saving_duration = time() - saving_time_start
                print("    Model parameters saved ({}).".format(
                    format_duration(saving_duration)))

                # Saving best model parameters yet
                if (validation_set
                        and lower_bound_valid > lower_bound_valid_maximum):
                    print(
                        "    Best validation lower_bound yet.",
                        "Saving model parameters as best model parameters."
                    )
                    saving_time_start = time()
                    lower_bound_valid_maximum = lower_bound_valid
                    current_checkpoint = (
                        tf.train.get_checkpoint_state(log_directory))
                    if current_checkpoint:
                        copy_model_directory(
                            current_checkpoint,
                            best_model_log_directory
                        )
                    remove_old_checkpoints(best_model_log_directory)
                    saving_duration = time() - saving_time_start
                    print("    Best model parameters saved ({}).".format(
                        format_duration(saving_duration)))

                print()

                # Plot latent validation values
                intermediate_analyser = kwargs.get("intermediate_analyser")
                if intermediate_analyser:

                    plotting_interval = kwargs.get("plotting_interval")
                    if plotting_interval is None:
                        plot_intermediate_results = (
                            epoch < 10
                            or epoch < 100 and (epoch + 1) % 10 == 0
                            or epoch < 1000 and (epoch + 1) % 50 == 0
                            or epoch > 1000 and (epoch + 1) % 100 == 0
                            or epoch == number_of_epochs - 1
                        )
                    else:
                        plot_intermediate_results = (
                            epoch % plotting_interval == 0)

                    if plot_intermediate_results:

                        if "mixture" in self.latent_distribution_name:
                            n_clusters = self.n_clusters
                            n_latent = self.latent_size
                            if "full-covariance" not in (
                                    self.latent_distribution_name):
                                p_z_covariances = numpy.empty(
                                    shape=[n_clusters, n_latent, n_latent])
                                q_z_covariances = numpy.empty(
                                    shape=[n_clusters, n_latent, n_latent])
                                for k in range(n_clusters):
                                    p_z_covariances[k] = numpy.diag(
                                        p_z_variances[k])
                                    q_z_covariances[k] = numpy.diag(
                                        q_z_variances[k])
                            centroids = {
                                "prior": {
                                    "probabilities": p_y_probabilities,
                                    "means": numpy.stack(p_z_means),
                                    "covariance_matrices": (
                                        p_z_covariances)
                                },
                                "posterior": {
                                    "probabilities": q_y_probabilities,
                                    "means": q_z_means,
                                    "covariance_matrices": (
                                        q_z_covariances)
                                }
                            }
                        else:
                            centroids = None

                        if validation_set:
                            intermediate_latent_values = z_mean_valid
                            intermediate_data_set = validation_set
                        else:
                            intermediate_latent_values = z_mean_train
                            intermediate_data_set = training_set

                        intermediate_analyser(
                            epoch=epoch,
                            learning_curves=learning_curves,
                            epoch_start=epoch_start,
                            model_type=self.type,
                            latent_values=intermediate_latent_values,
                            data_set=intermediate_data_set,
                            centroids=centroids,
                            model_name=self.name,
                            run_id=run_id,
                            analyses_directory=analyses_directory
                        )
                        print()
                    else:
                        intermediate_analyser(
                            epoch=epoch,
                            learning_curves=learning_curves,
                            epoch_start=epoch_start,
                            model_type=self.type,
                            model_name=self.name,
                            run_id=run_id,
                            analyses_directory=analyses_directory
                        )
                        print()

            training_duration = time() - training_time_start

            print("{} trained for {} epochs ({}).".format(
                capitalise_string(model_string),
                number_of_epochs,
                format_duration(training_duration))
            )
            print()

            # Clean up

            remove_old_checkpoints(log_directory)

            if temporary_log_directory:

                print("Moving log directory to permanent directory.")
                copying_time_start = time()

                if os.path.exists(permanent_log_directory):
                    shutil.rmtree(permanent_log_directory)

                shutil.move(log_directory, permanent_log_directory)

                copying_duration = time() - copying_time_start
                print("Log directory moved ({}).".format(format_duration(
                    copying_duration)))

                print()

            metadata_log["training duration"] = format_duration(
                training_duration)
            metadata_log["last epoch duration"] = format_duration(
                epoch_duration)

            metadata_log_filename = "metadata_log"
            epochs_trained = metadata_log.get("epochs trained")
            if epochs_trained:
                metadata_log_filename += "-" + "-".join(map(
                    str, epochs_trained))
            metadata_log_path = os.path.join(
                self.log_directory(run_id=run_id),
                metadata_log_filename + ".log"
            )
            with open(metadata_log_path, "w") as metadata_log_file:
                metadata_log_file.write("\n".join(
                    "{}: {}".format(metadata_field, metadata_value)
                    for metadata_field, metadata_value in metadata_log.items()
                    if metadata_value
                ))

            return 0

    def sample(self, sample_size=None, minibatch_size=None, run_id=None,
               use_early_stopping_model=False, use_best_model=False):
        """Sample from trained model.

        Arguments:
            sample_size (int, optional): The number of samples to draw
                from the model.
            minibatch_size (int, optional): The size of the random
                minibatches used at each step of training.
            run_id (str, optional): ID used to identify a certain run
                of the model.
            use_early_stopping_model (bool, optional): If ``True``, use
                model parameters, when early stopping triggered during
                training. Defaults to ``False``.
            use_best_model (bool, optional): If ``True``, use model
                parameters, which resulted in the best performance on
                validation set during training. Defaults to ``False``.

        Returns:
            A data set of generated examples/cells as well as a
            dictionary of data sets of samples for the two latent
            variables.
        """

        if sample_size is None:
            sample_size = defaults["models"]["sample_size"]
        if minibatch_size is None:
            minibatch_size = defaults["models"]["minibatch_size"]

        if run_id is None:
            run_id = defaults["models"]["run_id"]
        if run_id:
            run_id = check_run_id(run_id)
            model_string = "model for run {}".format(run_id)
        else:
            model_string = "model"

        batch_indices_samples = None
        if self.batch_correction:
            raise NotImplementedError("Sampling with batch correction.")

        if self.use_count_sum_as_parameter:
            raise NotImplementedError(
                "Sampling with count sum as reconstruction distribution "
                "parameter.")

        if self.use_count_sum_as_feature:
            raise NotImplementedError(
                "Sampling with count sum as additional latent feature.")

        log_directory = self.log_directory(
            run_id=run_id,
            early_stopping=use_early_stopping_model,
            best_model=use_best_model
        )

        checkpoint = tf.train.get_checkpoint_state(log_directory)

        with tf.Session(graph=self.graph) as session:

            if checkpoint:
                model_checkpoint_path = correct_model_checkpoint_path(
                    checkpoint.model_checkpoint_path,
                    log_directory
                )
                self.saver.restore(session, model_checkpoint_path)
            else:
                raise Exception(
                    "Cannot evaluate {} when it has not been trained.".format(
                        model_string)
                )

            print("Sampling {} examples from {}.".format(
                sample_size, model_string))
            sampling_time_start = time()

            y_samples = numpy.empty(
                shape=(sample_size, self.n_clusters),
                dtype=numpy.int32
            )
            z_mean_samples = numpy.empty(
                shape=(sample_size, self.latent_size),
                dtype=numpy.float32
            )
            x_mean = numpy.empty(
                shape=(sample_size, self.feature_size),
                dtype=numpy.float32
            )

            for i in range(0, sample_size, minibatch_size):

                indices = numpy.arange(
                    i, min(i + minibatch_size, sample_size))
                minibatch_sample_size = len(indices)

                (y_samples_i, z_mean_samples_i, z_samples_i) = session.run(
                    [
                        self.p_y_samples, self.p_z_mean_samples,
                        self.p_z_samples
                    ],
                    feed_dict={self.sample_size: minibatch_sample_size}
                )
                y_samples[indices] = y_samples_i
                z_mean_samples[indices] = z_mean_samples_i

                feed_dict_batch = {
                    self.y: y_samples_i,
                    self.is_training: False,
                    self.n_iw_samples: 1,
                    self.n_mc_samples: 1
                }
                feed_dict_batch.update({
                    self.z[k]: z_samples_i[k] for k in range(self.n_clusters)
                })

                # if self.batch_correction:
                #     feed_dict_batch[self.batch_indices] = (
                #         batch_indices_samples[indices])

                # if self.use_count_sum_as_parameter:
                #     feed_dict_batch[self.count_sum_parameter] = (
                #         count_sum_parameter_samples[indices])

                # if self.use_count_sum_as_feature:
                #     feed_dict_batch[self.count_sum_feature] = (
                #         count_sum_feature_samples[indices])

                x_mean_i = session.run(
                    self.p_x_mean,
                    feed_dict=feed_dict_batch
                )
                x_mean[indices] = x_mean_i

            sampling_duration = time() - sampling_time_start

            # Print evaluation
            print("Examples sampled ({}).".format(format_duration(
                sampling_duration)))

            # Data sets

            title = "Sampled data set"
            name = normalise_string(title)
            specifications = dict()
            sample_names = numpy.array([
                "sample {}".format(i + 1) for i in range(sample_size)])
            feature_names = numpy.array([
                "feature {}".format(i + 1) for i in range(self.feature_size)])

            sample_reconstruction_set = DataSet(
                name,
                title=title,
                specifications=specifications,
                values=x_mean,
                preprocessed_values=None,
                labels=None,
                example_names=sample_names,
                feature_names=feature_names,
                batch_indices=batch_indices_samples,
                feature_selection=None,
                example_filter=None,
                preprocessing_methods=None,
                kind="sample",
                version="reconstructed"
            )

            sample_z_set = DataSet(
                name,
                title=title,
                specifications=specifications,
                values=z_mean_samples,
                preprocessed_values=None,
                labels=None,
                example_names=sample_names,
                feature_names=numpy.array([
                    "z variable {}".format(i + 1)
                    for i in range(self.latent_size)
                ]),
                batch_indices=batch_indices_samples,
                feature_selection=None,
                example_filter=None,
                preprocessing_methods=None,
                kind="sample",
                version="z"
            )

            sample_y_set = DataSet(
                name,
                title=title,
                specifications=specifications,
                values=y_samples,
                preprocessed_values=None,
                labels=None,
                example_names=sample_names,
                feature_names=numpy.array([
                    "y variable {}".format(i + 1)
                    for i in range(self.n_clusters)
                ]),
                batch_indices=batch_indices_samples,
                feature_selection=None,
                example_filter=None,
                preprocessing_methods=None,
                kind="sample",
                version="y"
            )

            sample_latent_sets = {
                "z": sample_z_set,
                "y": sample_y_set
            }

            return sample_reconstruction_set, sample_latent_sets

    def evaluate(self, evaluation_set, minibatch_size=None, run_id=None,
                 use_early_stopping_model=False, use_best_model=False,
                 **kwargs):
        """Evaluate trained model

        Arguments:
            evaluation_set (DataSet): Data set used to evaluate model.
            minibatch_size (int, optional): The size of the random
                minibatches used at each step of training.
            run_id (str, optional): ID used to identify a certain run
                of the model.
            use_early_stopping_model (bool, optional): If ``True``, use
                model parameters, when early stopping triggered during
                training. Defaults to ``False``.
            use_best_model (bool, optional): If ``True``, use model
                parameters, which resulted in the best performance on
                validation set during training. Defaults to ``False``.

        Returns:
            A data set of reconstructed examples/cells as well as a
            dictionary of data sets of the two latent variables.
        """

        if minibatch_size is None:
            minibatch_size = defaults["models"]["minibatch_size"]

        if run_id is None:
            run_id = defaults["models"]["run_id"]
        if run_id:
            run_id = check_run_id(run_id)
            model_string = "model for run {}".format(run_id)
        else:
            model_string = "model"

        output_versions = kwargs.get("output_versions")
        if output_versions is None:
            output_versions = ["reconstructed", "latent"]
        elif output_versions == "all":
            output_versions = ["transformed", "reconstructed", "latent"]
        elif not isinstance(output_versions, list):
            output_versions = [output_versions]
        else:
            number_of_output_versions = len(output_versions)
            if number_of_output_versions > 3:
                raise ValueError(
                    "Can only output at most 3 sets, {} requested"
                    .format(number_of_output_versions)
                )
            elif number_of_output_versions != len(set(output_versions)):
                raise ValueError(
                    "Cannot output duplicate sets, {} requested."
                    .format(output_versions)
                )

        evaluation_subset_indices = kwargs.get("evaluation_subset_indices")
        if evaluation_subset_indices is None:
            evaluation_subset_indices = set()

        evaluation_set_transformed = False

        if self.batch_correction:
            batch_indices_eval = batch_indices_for_subset(evaluation_set)

        if self.use_count_sum_as_parameter:
            count_sum_parameter_eval = evaluation_set.count_sum

        if self.use_count_sum_as_feature:
            count_sum_feature_eval = evaluation_set.normalised_count_sum

        n_examples_eval = evaluation_set.number_of_examples
        n_feature_eval = evaluation_set.number_of_features

        noisy_preprocess = evaluation_set.noisy_preprocess

        if not noisy_preprocess:

            if evaluation_set.has_preprocessed_values:
                x_eval = evaluation_set.preprocessed_values
            else:
                x_eval = evaluation_set.values

            if self.reconstruction_distribution_name == "bernoulli":
                t_eval = evaluation_set.binarised_values
                evaluation_set_transformed = True
            else:
                t_eval = evaluation_set.values

        else:
            print("Noisily preprocess values.")
            noisy_time_start = time()
            x_eval = noisy_preprocess(evaluation_set.values)
            t_eval = x_eval
            evaluation_set_transformed = True
            noisy_duration = time() - noisy_time_start
            print("Values noisily preprocessed ({}).".format(
                format_duration(noisy_duration)))
            print()

        # Use label IDs instead of labels
        if evaluation_set.has_labels:

            class_names_to_class_ids = numpy.vectorize(
                lambda class_name:
                    evaluation_set.class_name_to_class_id[class_name]
            )
            class_ids_to_class_names = numpy.vectorize(
                lambda class_id:
                    evaluation_set.class_id_to_class_name[class_id]
            )

            evaluation_label_ids = class_names_to_class_ids(
                evaluation_set.labels)

            if evaluation_set.excluded_classes:
                excluded_class_ids = class_names_to_class_ids(
                    evaluation_set.excluded_classes)
            else:
                excluded_class_ids = []

        # Use superset label IDs instead of superset labels
        if evaluation_set.label_superset:

            superset_class_names_to_superset_class_ids = numpy.vectorize(
                lambda superset_class_name:
                    evaluation_set.superset_class_name_to_superset_class_id[
                        superset_class_name]
            )
            superset_class_ids_to_superset_class_names = numpy.vectorize(
                lambda superset_class_id:
                    evaluation_set.superset_class_id_to_superset_class_name[
                        superset_class_id]
            )

            evaluation_superset_label_ids = (
                superset_class_names_to_superset_class_ids(
                    evaluation_set.superset_labels))

            if evaluation_set.excluded_superset_classes:
                excluded_superset_class_ids = (
                    superset_class_names_to_superset_class_ids(
                        evaluation_set.excluded_superset_classes))
            else:
                excluded_superset_class_ids = []

        log_directory = self.log_directory(
            run_id=run_id,
            early_stopping=use_early_stopping_model,
            best_model=use_best_model
        )

        checkpoint = tf.train.get_checkpoint_state(log_directory)

        log_results = kwargs.get("log_results", True)
        if log_results:
            eval_summary_directory = os.path.join(log_directory, "evaluation")
            if os.path.exists(eval_summary_directory):
                shutil.rmtree(eval_summary_directory)

        with tf.Session(graph=self.graph) as session:

            if log_results:
                eval_summary_writer = tf.summary.FileWriter(
                    eval_summary_directory)

            if checkpoint:
                model_checkpoint_path = correct_model_checkpoint_path(
                    checkpoint.model_checkpoint_path,
                    log_directory
                )
                self.saver.restore(session, model_checkpoint_path)
                epoch = int(
                    os.path.split(model_checkpoint_path)[-1].split("-")[-1])
            else:
                raise Exception(
                    "Cannot evaluate {} when it has not been trained.".format(
                        model_string)
                )

            data_string = build_data_string(
                evaluation_set, self.reconstruction_distribution_name)
            print("Evaluating trained {} on {}.".format(
                model_string, data_string))
            evaluating_time_start = time()

            lower_bound_eval = 0
            kl_divergence_z_eval = 0
            kl_divergence_y_eval = 0
            reconstruction_error_eval = 0

            if log_results:
                q_y_probabilities = numpy.zeros(shape=self.n_clusters)
                q_z_means = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))
                q_z_variances = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))
                p_y_probabilities = numpy.zeros(shape=self.n_clusters)
                p_z_means = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))
                p_z_variances = numpy.zeros(
                    shape=(self.n_clusters, self.latent_size))
                if "full-covariance" in self.latent_distribution_name:
                    q_z_covariances = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size,
                               self.latent_size))
                    p_z_covariances = numpy.zeros(
                        shape=(self.n_clusters, self.latent_size,
                               self.latent_size))
                kl_divergence_z_neurons = numpy.zeros(shape=self.latent_size)

            q_y_logits = numpy.zeros(
                shape=(n_examples_eval, self.n_clusters))

            if "reconstructed" in output_versions:
                p_x_mean_eval = numpy.zeros(
                    shape=(n_examples_eval, n_feature_eval),
                    dtype=numpy.float32
                )
                p_x_stddev_eval = scipy.sparse.lil_matrix(
                    (n_examples_eval, n_feature_eval),
                    dtype=numpy.float32
                )
                stddev_of_p_x_given_z_mean_eval = scipy.sparse.lil_matrix(
                    (n_examples_eval, n_feature_eval),
                    dtype=numpy.float32
                )

            if "latent" in output_versions:
                z_mean_eval = numpy.zeros(
                    shape=(n_examples_eval, self.latent_size),
                    dtype=numpy.float32
                )
                y_mean_eval = numpy.zeros(
                    shape=(n_examples_eval, self.n_clusters),
                    dtype=numpy.float32
                )

            for i in range(0, n_examples_eval, minibatch_size):

                indices = numpy.arange(
                    i, min(i + minibatch_size, n_examples_eval))

                subset_indices = numpy.array(list(
                    evaluation_subset_indices.intersection(indices)))

                feed_dict_batch = {
                    self.x: x_eval[indices].toarray(),
                    self.t: t_eval[indices].toarray(),
                    self.is_training: False,
                    self.warm_up_weight: 1.0,
                    self.n_iw_samples:
                        self.number_of_importance_samples["evaluation"],
                    self.n_mc_samples:
                        self.number_of_monte_carlo_samples["evaluation"]
                }

                if self.batch_correction:
                    feed_dict_batch[self.batch_indices] = (
                        batch_indices_eval[indices])

                if self.use_count_sum_as_parameter:
                    feed_dict_batch[self.count_sum_parameter] = (
                        count_sum_parameter_eval[indices])

                if self.use_count_sum_as_feature:
                    feed_dict_batch[self.count_sum_feature] = (
                        count_sum_feature_eval[indices])

                (
                    lower_bound_i, reconstruction_error_i,
                    kl_divergence_z_i, kl_divergence_y_i,
                    q_y_probabilities_i, q_z_means_i, q_z_variances_i,
                    p_y_probabilities_i, p_z_means_i, p_z_variances_i,
                    q_z_covariances_i, p_z_covariances_i,
                    q_y_logits_i, p_x_mean_i,
                    p_x_stddev_i, stddev_of_p_x_given_z_mean_i,
                    y_mean_i, z_mean_i, kl_divergence_z_neurons_i
                ) = session.run(
                        [
                            self.lower_bound, self.reconstruction_error,
                            self.kl_divergence_z, self.kl_divergence_y,
                            self.q_y_probabilities, self.q_z_means,
                            self.q_z_variances, self.p_y_probabilities,
                            self.p_z_means, self.p_z_variances,
                            self.q_z_covariances, self.p_z_covariances,
                            self.q_y_logits, self.p_x_mean,
                            self.p_x_stddev, self.stddev_of_p_x_given_z_mean,
                            self.y_mean, self.z_mean,
                            self.kl_divergence_z_neurons
                        ],
                        feed_dict=feed_dict_batch
                    )

                lower_bound_eval += lower_bound_i
                kl_divergence_z_eval += kl_divergence_z_i
                kl_divergence_y_eval += kl_divergence_y_i
                reconstruction_error_eval += reconstruction_error_i

                if log_results:
                    q_y_probabilities += numpy.array(q_y_probabilities_i)
                    q_z_means += numpy.array(q_z_means_i)
                    q_z_variances += numpy.array(q_z_variances_i)
                    p_y_probabilities += numpy.array(p_y_probabilities_i)
                    p_z_means += numpy.array(p_z_means_i)
                    p_z_variances += numpy.array(p_z_variances_i)
                    if "full-covariance" in self.latent_distribution_name:
                        q_z_covariances += numpy.array(q_z_covariances_i)
                        p_z_covariances += numpy.array(p_z_covariances_i)
                    kl_divergence_z_neurons += numpy.array(
                        kl_divergence_z_neurons_i)

                q_y_logits[indices] = q_y_logits_i

                if "reconstructed" in output_versions:
                    p_x_mean_eval[indices] = p_x_mean_i

                    if subset_indices.size > 0:
                        p_x_stddev_eval[subset_indices] = (
                            p_x_stddev_i[subset_indices - i])
                        stddev_of_p_x_given_z_mean_eval[subset_indices] = (
                            stddev_of_p_x_given_z_mean_i[subset_indices - i])

                if "latent" in output_versions:
                    y_mean_eval[indices] = y_mean_i
                    z_mean_eval[indices] = z_mean_i

            lower_bound_eval /= n_examples_eval / minibatch_size
            kl_divergence_z_eval /= n_examples_eval / minibatch_size
            kl_divergence_y_eval /= n_examples_eval / minibatch_size
            reconstruction_error_eval /= n_examples_eval / minibatch_size

            if log_results:
                q_y_probabilities /= n_examples_eval / minibatch_size
                q_z_means /= n_examples_eval / minibatch_size
                q_z_variances /= n_examples_eval / minibatch_size
                p_y_probabilities /= n_examples_eval / minibatch_size
                p_z_means /= n_examples_eval / minibatch_size
                p_z_variances /= n_examples_eval / minibatch_size
                if "full-covariance" in self.latent_distribution_name:
                    q_z_covariances /= n_examples_eval / minibatch_size
                    p_z_covariances /= n_examples_eval / minibatch_size
                kl_divergence_z_neurons /= n_examples_eval / minibatch_size

            if (self.number_of_importance_samples["evaluation"] == 1
                    and self.number_of_monte_carlo_samples["evaluation"] == 1):
                stddev_of_p_x_given_z_mean_eval = None

            evaluation_cluster_ids = q_y_logits.argmax(axis=1)

            if evaluation_set.has_labels:
                predicted_evaluation_label_ids = map_cluster_ids_to_label_ids(
                    evaluation_label_ids,
                    evaluation_cluster_ids,
                    excluded_class_ids
                )
                accuracy_eval = accuracy(
                    evaluation_label_ids,
                    predicted_evaluation_label_ids,
                    excluded_class_ids
                )
            else:
                accuracy_eval = None

            if evaluation_set.label_superset:
                predicted_evaluation_superset_label_ids = (
                    map_cluster_ids_to_label_ids(
                        evaluation_superset_label_ids,
                        evaluation_cluster_ids,
                        excluded_superset_class_ids
                    )
                )
                accuracy_superset_eval = accuracy(
                    evaluation_superset_label_ids,
                    predicted_evaluation_superset_label_ids,
                    excluded_superset_class_ids
                )
                accuracy_display = accuracy_superset_eval
            else:
                accuracy_superset_eval = None
                accuracy_display = accuracy_eval

            if log_results:

                summary = tf.Summary()
                summary.value.add(
                    tag="losses/lower_bound",
                    simple_value=lower_bound_eval
                )
                summary.value.add(
                    tag="losses/reconstruction_error",
                    simple_value=reconstruction_error_eval
                )
                summary.value.add(
                    tag="losses/kl_divergence_z",
                    simple_value=kl_divergence_z_eval
                )
                summary.value.add(
                    tag="losses/kl_divergence_y",
                    simple_value=kl_divergence_y_eval
                )
                summary.value.add(tag="accuracy", simple_value=accuracy_eval)
                if accuracy_superset_eval:
                    summary.value.add(
                        tag="superset_accuracy",
                        simple_value=accuracy_superset_eval
                    )

                for k in range(self.n_clusters):
                    summary.value.add(
                        tag="prior/cluster_{}/probability".format(k),
                        simple_value=p_y_probabilities[k]
                    )
                    summary.value.add(
                        tag="posterior/cluster_{}/probability".format(k),
                        simple_value=q_y_probabilities[k]
                    )
                    for l in range(self.latent_size):
                        summary.value.add(
                            tag="prior/cluster_{}/mean/dimension_{}".format(
                                k, l),
                            simple_value=p_z_means[k][l]
                        )
                        summary.value.add(
                            tag="posterior/cluster_{}/mean/dimension_{}"
                                .format(k, l),
                            simple_value=q_z_means[k, l]
                        )
                        summary.value.add(
                            tag="prior/cluster_{}/variance/dimension_{}"
                                .format(k, l),
                            simple_value=p_z_variances[k][l]
                        )
                        summary.value.add(
                            tag="posterior/cluster_{}/variance/dimension_{}"
                                .format(k, l),
                            simple_value=q_z_variances[k, l]
                        )
                        if "full-covariance" in self.latent_distribution_name:
                            for l_ in range(self.latent_size):
                                summary.value.add(
                                    tag="prior/cluster_{}/covariance/"
                                        "dimension_{}_{}".format(k, l, l_),
                                    simple_value=p_z_covariances[k, l, l_]
                                )
                                summary.value.add(
                                    tag="posterior/cluster_{}/covariance/"
                                        "dimension_{}_{}".format(k, l, l_),
                                    simple_value=q_z_covariances[k, l, l_]
                                )
                for l in range(self.latent_size):
                    summary.value.add(
                        tag="kl_divergence_neurons/{}".format(l),
                        simple_value=kl_divergence_z_neurons[l]
                    )

                eval_summary_writer.add_summary(
                    summary, global_step=epoch)
                eval_summary_writer.flush()

            evaluating_duration = time() - evaluating_time_start

            evaluation_string = "    {} set ({}): ".format(
                evaluation_set.kind.capitalize(),
                format_duration(evaluating_duration))
            evaluation_metrics = [
                "ELBO: {:.5g}".format(lower_bound_eval),
                "ENRE: {:.5g}".format(reconstruction_error_eval),
                "KL_z: {:.5g}".format(kl_divergence_z_eval),
                "KL_y: {:.5g}".format(kl_divergence_y_eval)
            ]
            if accuracy_display:
                evaluation_metrics.append(
                    "Acc: {:.5g}".format(accuracy_display)
                )
            evaluation_string += ", ".join(evaluation_metrics)
            evaluation_string += "."

            print(evaluation_string)

            # Data sets
            output_sets = [None] * len(output_versions)

            if "transformed" in output_versions:
                if evaluation_set_transformed:
                    transformed_evaluation_set = DataSet(
                        evaluation_set.name,
                        title=evaluation_set.title,
                        specifications=evaluation_set.specifications,
                        values=t_eval,
                        preprocessed_values=None,
                        labels=evaluation_set.labels,
                        example_names=evaluation_set.example_names,
                        feature_names=evaluation_set.feature_names,
                        batch_indices=evaluation_set.batch_indices,
                        batch_names=evaluation_set.batch_names,
                        features_mapped=evaluation_set.features_mapped,
                        feature_selection=evaluation_set.feature_selection,
                        example_filter=evaluation_set.example_filter,
                        preprocessing_methods=(
                            evaluation_set.preprocessing_methods),
                        kind=evaluation_set.kind,
                        version="transformed"
                    )
                else:
                    transformed_evaluation_set = evaluation_set

                index = output_versions.index("transformed")
                output_sets[index] = transformed_evaluation_set

            if "reconstructed" in output_versions:
                reconstructed_evaluation_set = DataSet(
                    evaluation_set.name,
                    title=evaluation_set.title,
                    specifications=evaluation_set.specifications,
                    values=p_x_mean_eval,
                    total_standard_deviations=p_x_stddev_eval,
                    explained_standard_deviations=(
                        stddev_of_p_x_given_z_mean_eval),
                    preprocessed_values=None,
                    labels=evaluation_set.labels,
                    example_names=evaluation_set.example_names,
                    feature_names=evaluation_set.feature_names,
                    batch_indices=evaluation_set.batch_indices,
                    batch_names=evaluation_set.batch_names,
                    features_mapped=evaluation_set.features_mapped,
                    feature_selection=evaluation_set.feature_selection,
                    example_filter=evaluation_set.example_filter,
                    preprocessing_methods=evaluation_set.preprocessing_methods,
                    kind=evaluation_set.kind,
                    version="reconstructed"
                )
                index = output_versions.index("reconstructed")
                output_sets[index] = reconstructed_evaluation_set

            if "latent" in output_versions:
                z_evaluation_set = DataSet(
                    evaluation_set.name,
                    title=evaluation_set.title,
                    specifications=evaluation_set.specifications,
                    values=z_mean_eval,
                    preprocessed_values=None,
                    labels=evaluation_set.labels,
                    example_names=evaluation_set.example_names,
                    feature_names=numpy.array([
                        "z variable {}".format(i + 1)
                        for i in range(self.latent_size)
                    ]),
                    batch_indices=evaluation_set.batch_indices,
                    batch_names=evaluation_set.batch_names,
                    features_mapped=evaluation_set.features_mapped,
                    feature_selection=evaluation_set.feature_selection,
                    example_filter=evaluation_set.example_filter,
                    preprocessing_methods=evaluation_set.preprocessing_methods,
                    kind=evaluation_set.kind,
                    version="z"
                )

                y_evaluation_set = DataSet(
                    evaluation_set.name,
                    title=evaluation_set.title,
                    specifications=evaluation_set.specifications,
                    values=y_mean_eval,
                    preprocessed_values=None,
                    labels=evaluation_set.labels,
                    example_names=evaluation_set.example_names,
                    feature_names=numpy.array([
                        "y variable {}".format(i + 1)
                        for i in range(self.n_clusters)
                    ]),
                    batch_indices=evaluation_set.batch_indices,
                    batch_names=evaluation_set.batch_names,
                    features_mapped=evaluation_set.features_mapped,
                    feature_selection=evaluation_set.feature_selection,
                    example_filter=evaluation_set.example_filter,
                    preprocessing_methods=evaluation_set.preprocessing_methods,
                    kind=evaluation_set.kind,
                    version="y"
                )

                latent_evaluation_sets = {
                    "z": z_evaluation_set,
                    "y": y_evaluation_set
                }

                index = output_versions.index("latent")
                output_sets[index] = latent_evaluation_sets

            prediction_specifications = PredictionSpecifications(
                method="model",
                number_of_clusters=self.n_clusters,
                training_set_kind=None
            )

            if evaluation_set.has_labels:
                predicted_evaluation_labels = class_ids_to_class_names(
                    predicted_evaluation_label_ids)
            else:
                predicted_evaluation_labels = None

            if evaluation_set.has_superset_labels:
                predicted_evaluation_superset_labels = (
                    superset_class_ids_to_superset_class_names(
                        predicted_evaluation_superset_label_ids))
            else:
                predicted_evaluation_superset_labels = None

            def update_predictions(subset):
                subset.update_predictions(
                    prediction_specifications=prediction_specifications,
                    predicted_cluster_ids=evaluation_cluster_ids,
                    predicted_labels=predicted_evaluation_labels,
                    predicted_superset_labels=(
                        predicted_evaluation_superset_labels)
                )

            for output_set in output_sets:
                if isinstance(output_set, dict):
                    for variable in output_set:
                        update_predictions(output_set[variable])
                else:
                    update_predictions(output_set)

            if len(output_sets) == 1:
                output_sets = output_sets[0]

            return output_sets

    def _setup_model_graph(self):
        # Retrieving layers parameterising all distributions in model:

        # Y latent space
        with tf.variable_scope("Y"):
            # p(y) = Cat(pi)
            # Shape: (1, K), so first minibatch dimension can be broadcast to y
            with tf.variable_scope("P"):
                if self.prior_probabilities_method == "custom":
                    p_y_probabilities = tf.constant(self.prior_probabilities)
                    p_y_logits = tf.log(p_y_probabilities)
                elif self.prior_probabilities_method == "learn":
                    p_y_logits = tf.Variable(
                        initial_value=tf.zeros(shape=(self.n_clusters)),
                        name="LOGITS"
                    )
                elif self.prior_probabilities_method == "uniform":
                    p_y_probabilities = (
                        tf.ones(self.n_clusters) / self.n_clusters)
                    p_y_logits = tf.log(p_y_probabilities)

                p_y_logits = tf.reshape(
                    p_y_logits,
                    shape=[1, self.n_clusters]
                )
                self.p_y = tfp.distributions.Categorical(logits=p_y_logits)
                self.p_y_probabilities = tf.reshape(
                    self.p_y.probs, shape=[self.n_clusters])

                p_y_samples = self.p_y.sample(sample_shape=(
                    self.sample_size))
                self.p_y_samples = tf.squeeze(
                    tf.one_hot(
                        indices=p_y_samples,
                        depth=self.n_clusters
                    ),
                    axis=1
                )

            # q(y|x) = Cat(pi(x))
            identity_matrix = numpy.eye(self.n_clusters)
            y = [
                tf.reshape(
                    tf.constant(
                        identity_matrix[k],
                        name="hot_at_{:d}".format(k),
                        dtype=tf.float32
                    ),
                    shape=(-1, self.n_clusters)
                )
                for k in range(self.n_clusters)
            ]

            self.q_y_given_x = self._build_graph_for_q_y_given_x(self.x)
            self.q_y_logits = self.q_y_given_x.logits
            self.q_y_probabilities = tf.reduce_mean(self.q_y_given_x.probs, 0)

        # z latent space
        with tf.variable_scope("Z"):
            self.q_z_given_x_y = [None]*self.n_clusters
            z_mean = [None]*self.n_clusters
            self.z = [None]*self.n_clusters
            self.p_z_given_y = [None]*self.n_clusters
            self.p_z_mean = [None]*self.n_clusters
            self.p_z_means = []
            self.p_z_variances = []
            self.p_z_covariances = []
            self.q_z_means = []
            self.q_z_variances = []
            self.q_z_covariances = []
            # Loop over parameter layers for all K gaussians.
            for k in range(self.n_clusters):
                if k >= 1:
                    reuse_weights = True
                else:
                    reuse_weights = False

                # Latent prior distribution
                self.q_z_given_x_y[k], z_mean[k], self.z[k] = (
                    self._build_graph_for_q_z_given_x_y(
                        self.x, y[k],
                        distribution_name=self.latent_distribution[
                            "z posterior"],
                        reuse=reuse_weights))
                # Latent prior distribution
                self.p_z_given_y[k], self.p_z_mean[k] = (
                    self._build_graph_for_p_z_given_y(
                        y[k],
                        distribution_name=self.latent_distribution["z prior"],
                        reuse=reuse_weights))

                self.p_z_means.append(
                    tf.reduce_mean(self.p_z_given_y[k].mean(), [0, 1, 2]))
                self.p_z_variances.append(tf.square(tf.reduce_mean(
                    self.p_z_given_y[k].stddev(), axis=[0, 1, 2])))

                self.q_z_means.append(tf.reduce_mean(
                    self.q_z_given_x_y[k].mean(), axis=[0, 1, 2]))
                self.q_z_variances.append(tf.reduce_mean(
                    tf.square(self.q_z_given_x_y[k].stddev()), axis=[0, 1, 2]))

                if "full-covariance" in self.latent_distribution_name:
                    self.p_z_covariances.append(tf.reduce_mean(
                        self.p_z_given_y[k].covariance(), axis=[0, 1, 2]))
                    self.q_z_covariances.append(tf.reduce_mean(
                        self.q_z_given_x_y[k].covariance(), axis=[0, 1, 2]))

            self.y = self.q_y_given_x.probs
            self.z_mean = tf.add_n([
                z_mean[k] * tf.expand_dims(self.y[:, k], -1)
                for k in range(self.n_clusters)
            ])

            self.p_z_samples = [
                tf.squeeze(
                    self.p_z_given_y[k].sample(sample_shape=(self.sample_size))
                ) for k in range(self.n_clusters)
            ]
            self.p_z_mean_samples = tf.add_n([
                self.p_z_samples[k] * tf.expand_dims(
                    self.p_y_samples[:, k], -1)
                for k in range(self.n_clusters)
            ])

        # Decoder for x
        with tf.variable_scope("X"):
            self.p_x_given_z = [None]*self.n_clusters
            for k in range(self.n_clusters):
                if k >= 1:
                    reuse_weights = True
                else:
                    reuse_weights = False

                self.p_x_given_z[k] = self._build_graph_for_p_x_given_z(
                    self.z[k], reuse=reuse_weights)

        # (B, K)
        self.y_mean = self.y
        # (R, L, Bs, K)
        self.q_y_logits = tf.reshape(
            self.q_y_given_x.logits, shape=[1, -1, self.n_clusters])

        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            parameter_summary = tf.summary.histogram(parameter.name, parameter)
            self.parameter_summary_list.append(parameter_summary)
        self.parameter_summary = tf.summary.merge(self.parameter_summary_list)

    def _build_graph_for_q_z_given_x_y(
            self, x, y, distribution_name="softplus gaussian", reuse=False):

        # Encoder for q(z|x,y_i=1) = N(mu(x,y_i=1), sigma^2(x,y_i=1))
        with tf.variable_scope("Q"):
            distribution = DISTRIBUTIONS[distribution_name]
            y = tf.broadcast_to(y, shape=(tf.shape(self.x)[0], tf.shape(y)[1]))
            xy = tf.concat((self.x, y), axis=-1)
            encoder = dense_layers(
                inputs=xy,
                num_outputs=self.hidden_sizes,
                activation_fn=tf.nn.relu,
                minibatch_normalisation=self.minibatch_normalisation,
                is_training=self.is_training,
                input_dropout_keep_probability=self.dropout_keep_probability_x,
                hidden_dropout_keep_probability=(
                    self.dropout_keep_probability_h),
                scope="ENCODER",
                layer_name="LAYER",
                reuse=reuse
            )

            with tf.variable_scope(normalise_string(
                    distribution_name).upper()):
                # Loop over and add parameter layers to theta dict.
                theta = {}
                for parameter in distribution["parameters"]:
                    parameter_activation_function = distribution["parameters"][
                        parameter]["activation function"]
                    p_min, p_max = distribution["parameters"][parameter][
                        "support"]
                    n_outputs = self.latent_size
                    size_function = distribution["parameters"][parameter].get(
                        "size function")
                    if size_function:
                        n_outputs = size_function(self.latent_size)
                    theta[parameter] = tf.expand_dims(tf.expand_dims(
                        dense_layer(
                            inputs=encoder,
                            num_outputs=n_outputs,
                            activation_fn=lambda x: tf.clip_by_value(
                                parameter_activation_function(x),
                                p_min + numpy.finfo(dtype=numpy.float32).tiny,
                                p_max - numpy.finfo(dtype=numpy.float32).tiny
                            ),
                            is_training=self.is_training,
                            dropout_keep_probability=(
                                self.dropout_keep_probability_h),
                            scope=parameter.upper(),
                            reuse=reuse
                        ), axis=0), axis=0
                    )

                # Parameterise
                q_z_given_x_y = distribution["class"](theta)

                # Analytical mean
                z_mean = q_z_given_x_y.mean()

                # Sampling of importance weighting and Monte Carlo samples
                z_samples = q_z_given_x_y.sample(
                    self.n_iw_samples * self.n_mc_samples)
                z = tf.cast(
                    tf.reshape(
                        z_samples,
                        shape=[-1, self.latent_size],
                        name="SAMPLES"
                    ),
                    dtype=tf.float32
                )

        return q_z_given_x_y, z_mean, z

    def _build_graph_for_p_z_given_y(
            self, y, distribution_name="softplus gaussian", reuse=False):

        with tf.variable_scope("P"):
            with tf.variable_scope(normalise_string(
                    distribution_name).upper()):
                distribution = DISTRIBUTIONS[distribution_name]
                # Loop over and add parameter layers to theta dict.
                theta = {}
                for parameter in distribution["parameters"]:
                    parameter_activation_function = distribution["parameters"][
                        parameter]["activation function"]
                    p_min, p_max = distribution["parameters"][parameter][
                        "support"]
                    n_outputs = self.latent_size
                    size_function = distribution["parameters"][parameter].get(
                        "size function")
                    if size_function:
                        n_outputs = size_function(self.latent_size)
                    theta[parameter] = tf.expand_dims(tf.expand_dims(
                        dense_layer(
                            inputs=y,
                            num_outputs=n_outputs,
                            activation_fn=lambda x: tf.clip_by_value(
                                parameter_activation_function(x),
                                p_min + numpy.finfo(dtype=numpy.float32).tiny,
                                p_max - numpy.finfo(dtype=numpy.float32).tiny
                            ),
                            is_training=self.is_training,
                            dropout_keep_probability=(
                                self.dropout_keep_probability_y),
                            scope=parameter.upper(),
                            reuse=reuse
                        ), axis=0), axis=0
                    )

                p_z_given_y = distribution["class"](theta)
                p_z_mean = tf.reduce_mean(p_z_given_y.mean())

        return p_z_given_y, p_z_mean

    def _build_graph_for_q_y_given_x(
            self, x, distribution_name="categorical", reuse=False):

        with tf.variable_scope(distribution_name.upper()):
            distribution = DISTRIBUTIONS[distribution_name]
            # Encoder
            encoder = dense_layers(
                inputs=self.x,
                num_outputs=self.hidden_sizes,
                activation_fn=tf.nn.relu,
                minibatch_normalisation=self.minibatch_normalisation,
                is_training=self.is_training,
                input_dropout_keep_probability=self.dropout_keep_probability_x,
                hidden_dropout_keep_probability=(
                    self.dropout_keep_probability_h),
                scope="ENCODER",
                layer_name="LAYER",
                reuse=reuse
            )
            # Loop over and add parameter layers to theta dict.
            theta = {}
            for parameter in distribution["parameters"]:
                parameter_activation_function = distribution["parameters"][
                    parameter]["activation function"]
                p_min, p_max = distribution["parameters"][parameter]["support"]
                theta[parameter] = dense_layer(
                    inputs=encoder,
                    num_outputs=self.n_clusters,
                    activation_fn=lambda x: tf.clip_by_value(
                        parameter_activation_function(x),
                        p_min + numpy.finfo(dtype=numpy.float32).tiny,
                        p_max - numpy.finfo(dtype=numpy.float32).tiny
                    ),
                    is_training=self.is_training,
                    dropout_keep_probability=(
                        self.dropout_keep_probability_h),
                    scope=parameter.upper(),
                    reuse=reuse
                )
            # Parameterise q(y|x) = Cat(pi(x))
            q_y_given_x = distribution["class"](theta)

        return q_y_given_x

    def _build_graph_for_p_x_given_z(self, z, reuse=False):
        # Decoder - Generative model, p(x|z)

        decoder_inputs = [z]

        # Make sure we use a replication per sample of the feature sum,
        # when adding this to the features
        if self.batch_correction:
            batch_indices_one_hot = tf.squeeze(
                tf.one_hot(
                    indices=self.batch_indices,
                    depth=self.number_of_batches
                ),
                axis=1
            )
            replicated_batch_indices = tf.tile(
                batch_indices_one_hot,
                multiples=[self.n_iw_samples * self.n_mc_samples, 1],
                name="BATCH_INDICES"
            )
            decoder_inputs.append(replicated_batch_indices)

        # Make sure we use a replication per sample of the feature sum,
        # when adding this to the features
        if self.use_count_sum_as_feature:
            replicated_count_sum_feature = tf.tile(
                self.count_sum_feature,
                multiples=[self.n_iw_samples * self.n_mc_samples, 1],
                name="COUNT_SUM"
            )
            decoder_inputs.append(replicated_count_sum_feature)

        if len(decoder_inputs) > 1:
            decoder = tf.concat(
                decoder_inputs,
                axis=-1,
                name="-".join(i.name.replace(":0", "") for i in decoder_inputs)
            )
        else:
            decoder = z

        decoder = dense_layers(
            inputs=decoder,
            num_outputs=self.hidden_sizes[::-1],
            activation_fn=tf.nn.relu,
            minibatch_normalisation=self.minibatch_normalisation,
            is_training=self.is_training,
            input_dropout_keep_probability=self.dropout_keep_probability_z,
            hidden_dropout_keep_probability=self.dropout_keep_probability_h,
            scope="DECODER",
            layer_name="LAYER",
            reuse=reuse
        )

        # Reconstruction distribution parameterisation

        with tf.variable_scope("DISTRIBUTION"):

            x_theta = {}

            for parameter in self.reconstruction_distribution["parameters"]:

                parameter_activation_function = (
                    self.reconstruction_distribution["parameters"][parameter][
                        "activation function"])
                p_min, p_max = self.reconstruction_distribution["parameters"][
                    parameter]["support"]

                x_theta[parameter] = dense_layer(
                    inputs=decoder,
                    num_outputs=self.feature_size,
                    activation_fn=lambda x: tf.clip_by_value(
                        parameter_activation_function(x),
                        p_min + numpy.finfo(dtype=numpy.float32).tiny,
                        p_max - numpy.finfo(dtype=numpy.float32).tiny
                    ),
                    is_training=self.is_training,
                    dropout_keep_probability=self.dropout_keep_probability_h,
                    scope=parameter.upper(),
                    reuse=reuse
                )

            if ("constrained" in self.reconstruction_distribution_name
                    or "multinomial" in self.reconstruction_distribution_name):
                p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta,
                    self.replicated_count_sum_parameter
                )
            elif "multinomial" in self.reconstruction_distribution_name:
                p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta,
                    self.replicated_count_sum_parameter
                )
            else:
                p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta
                )

            if self.k_max:
                x_logits = dense_layer(
                    inputs=decoder,
                    num_outputs=(
                        self.feature_size
                        * self.number_of_reconstruction_classes
                    ),
                    activation_fn=None,
                    is_training=self.is_training,
                    dropout_keep_probability=self.dropout_keep_probability_h,
                    scope="P_K",
                    reuse=reuse
                )

                x_logits = tf.reshape(
                    x_logits,
                    shape=[
                        -1,
                        self.feature_size,
                        self.number_of_reconstruction_classes
                    ]
                )

                p_x_given_z = Categorised(
                    dist=p_x_given_z,
                    cat=tfp.distributions.Categorical(
                        logits=x_logits)
                )

            return p_x_given_z

    def _setup_loss_function(self):
        # Prepare replicated and reshaped arrays
        # Replicate out minibatches in tiles per sample into shape
        # (R * L * B, N_x)
        t_tiled = tf.tile(self.t, [self.n_iw_samples*self.n_mc_samples, 1])
        # Reshape samples back to shape (R, L, B, N_z)
        z_reshaped = [
            tf.reshape(
                self.z[k],
                shape=[
                    self.n_iw_samples,
                    self.n_mc_samples,
                    -1,
                    self.latent_size
                ]
            )
            for k in range(self.n_clusters)
        ]

        if self.prior_probabilities_method == "uniform":
            # H[q(y|x)] = -E_{q(y|x)}[ log(q(y|x)) ]
            # Shape: (B)
            q_y_given_x_entropy = self.q_y_given_x.entropy()
            # H[q(y|x)||p(y)] = -E_{q(y|x)}[log(p(y))] = -E_{q(y|x)}[ log(1/K)]
            #                 = log(K)
            # Shape: ()
            p_y_entropy = numpy.log(self.n_clusters)
            # KL(q||p) = -E_q(y|x)[log p(y)/q(y|x)]
            #          = -E_q(y|x)[log p(y)] + E_q(y|x)[log q(y|x)]
            #          = H(q|p) - H(q)
            # Shape: (B)
            kl_divergence_y = p_y_entropy - q_y_given_x_entropy
        else:
            kl_divergence_y = tfp.distributions.kl_divergence(
                self.q_y_given_x, self.p_y)
            p_y_entropy = tf.squeeze(self.p_y.entropy())

        kl_divergence_y_threshhold = (
            self.proportion_of_free_nats_for_y_kl_divergence * p_y_entropy)

        kl_divergence_z = [None] * self.n_clusters
        kl_divergence_z_mean = [None] * self.n_clusters
        log_p_x_given_z_mean = [None] * self.n_clusters
        p_x_means = [None] * self.n_clusters
        mean_of_p_x_given_z_variances = [None] * self.n_clusters
        variance_of_p_x_given_z_means = [None] * self.n_clusters

        for k in range(self.n_clusters):
            # (R, L, B, L) --> (R, L, B)
            log_q_z_given_x_y = tf.reduce_sum(
                self.q_z_given_x_y[k].log_prob(
                    z_reshaped[k]
                ),
                axis=-1
            )
            # (R, L, B, L) --> (R, L, B)
            log_p_z_given_y = tf.reduce_sum(
                self.p_z_given_y[k].log_prob(
                    z_reshaped[k]
                ),
                axis=-1
            )
            # (R, L, B)
            kl_divergence_z[k] = log_q_z_given_x_y - log_p_z_given_y

            # (R, L, B) --> (B)
            kl_divergence_z_mean[k] = tf.reduce_mean(
                kl_divergence_z[k],
                axis=(0, 1)
            ) * self.y[:, k]

            # (R, L, B, F)
            p_x_given_z_log_prob = self.p_x_given_z[k].log_prob(t_tiled)

            # (R, L, B, F) --> (R, L, B)
            log_p_x_given_z = tf.reshape(
                tf.reduce_sum(
                    p_x_given_z_log_prob,
                    axis=-1
                ),
                shape=[self.n_iw_samples, self.n_mc_samples, -1]
            )
            # (R, L, B) --> (B)
            log_p_x_given_z_mean[k] = tf.reduce_mean(
                log_p_x_given_z,
                axis=(0, 1)
            ) * self.y[:, k]

            # (R * L * B, F) --> (R, L, B, F)
            p_x_given_z_mean = tf.reshape(
                self.p_x_given_z[k].mean(),
                shape=[
                    self.n_iw_samples,
                    self.n_mc_samples,
                    -1,
                    self.feature_size
                ]
            )

            # (R, L, B, F) --> (R, B, F) --> (B, F)
            p_x_means[k] = tf.reduce_mean(
                tf.reduce_mean(
                    p_x_given_z_mean,
                    axis=1
                ),
                axis=0
            ) * tf.expand_dims(self.y[:, k], -1)

            # Reconstruction standard deviation:
            #      sqrt(V[x]) = sqrt(E[V[x|z]] + V[E[x|z]])
            #      = E_z[p_x_given_z.var] + E_z[(p_x_given_z.mean - E[x])^2]

            # Ê[V[x|z]] \approx q(y|x) * 1/(R*L) \sum^R_r w_r \sum^L_{l=1}
            #                 * E[x|z_lr]
            # (R * L * B, F) --> (R, L, B, F) --> (R, B, F) --> (B, F)
            mean_of_p_x_given_z_variances[k] = tf.reduce_mean(
                tf.reduce_mean(
                    tf.reshape(
                        self.p_x_given_z[k].variance(),
                        shape=[
                            self.n_iw_samples,
                            self.n_mc_samples,
                            -1,
                            self.feature_size
                        ]
                    ),
                    axis=1
                ),
                axis=0
            ) * tf.expand_dims(self.y[:, k], -1)

            # Estimated variance of likelihood expectation:
            # ^V[E[x|z]] = ( E[x|z_l] - Ê[x] )^2
            # (R, L, B, F)
            variance_of_p_x_given_z_means[k] = tf.reduce_mean(
                tf.reduce_mean(
                    tf.square(
                        p_x_given_z_mean - tf.reshape(
                            p_x_means[k],
                            shape=[1, 1, -1, self.feature_size]
                        )
                    ),
                    axis=1
                ),
                axis=0
            ) * tf.expand_dims(self.y[:, k], -1)

        # Marginalise y out in list by add_n and reshape from
        # K * [(B, F)] --> (B, F)
        self.variance_of_p_x_given_z_mean = tf.add_n(
            variance_of_p_x_given_z_means
        )
        self.mean_of_p_x_given_z_variance = tf.add_n(
            mean_of_p_x_given_z_variances
        )
        self.p_x_stddev = tf.sqrt(
            self.mean_of_p_x_given_z_variance
            + self.variance_of_p_x_given_z_mean
        )
        self.stddev_of_p_x_given_z_mean = tf.sqrt(
            self.variance_of_p_x_given_z_mean
        )

        self.p_x_mean = tf.add_n(p_x_means)

        # (B) --> ()
        self.kl_divergence_z = tf.reduce_mean(tf.add_n(kl_divergence_z_mean))
        self.kl_divergence_y = tf.reduce_mean(kl_divergence_y)
        if self.proportion_of_free_nats_for_y_kl_divergence:
            kl_divergence_y_modified = tf.where(
                self.kl_divergence_y > kl_divergence_y_threshhold,
                self.kl_divergence_y,
                kl_divergence_y_threshhold
            )
        else:
            kl_divergence_y_modified = self.kl_divergence_y

        self.kl_divergence = self.kl_divergence_z + self.kl_divergence_y
        self.kl_divergence_neurons = tf.expand_dims(self.kl_divergence, -1)
        self.reconstruction_error = tf.reduce_mean(tf.add_n(
            log_p_x_given_z_mean))
        self.lower_bound = self.reconstruction_error - self.kl_divergence
        self.lower_bound_weighted = (
            self.reconstruction_error
            - self.warm_up_weight * self.kl_weight * (
                self.kl_divergence_z + kl_divergence_y_modified
            )
        )

        kl_divergence_z_neurons = [None] * self.n_clusters
        kl_divergence_z_neurons_mean = [None] * self.n_clusters

        for k in range(self.n_clusters):
            # (R, I, B, L)
            log_q_z_given_x_y = self.q_z_given_x_y[k].log_prob(z_reshaped[k])

            # (R, I, B, L)
            log_p_z_given_y = self.p_z_given_y[k].log_prob(z_reshaped[k])

            # (R, I, B, L)
            kl_divergence_z_neurons[k] = log_q_z_given_x_y - log_p_z_given_y

            # (R, I, B, L) --> (B, L)
            kl_divergence_z_neurons_mean[k] = tf.reduce_mean(
                kl_divergence_z_neurons[k],
                axis=(0, 1)
            ) * tf.expand_dims(self.y[:, k], -1)
            # ) * tf.tile(self.y[:, k], [self.latent_size])

        self.kl_divergence_z_neurons = tf.reduce_mean(
            tf.add_n(kl_divergence_z_neurons_mean),
            axis=0)

    def _setup_optimiser(self):

        # Create the gradient descent optimiser with the given learning rate.
        def _optimiser():

            # Optimiser and training objective of negative loss
            optimiser = tf.train.AdamOptimizer(self.learning_rate)

            # Create a variable to track the global step.
            self.global_step = tf.Variable(
                0,
                name="global_step",
                trainable=False
            )

            gradients = optimiser.compute_gradients(-self.lower_bound_weighted)
            clipped_gradients = [
                (tf.clip_by_value(gradient, -1., 1.), variable)
                for gradient, variable in gradients
            ]
            self.optimiser = optimiser.apply_gradients(
                clipped_gradients,
                global_step=self.global_step
            )

        # Make sure that the updates of the moving_averages in minibatch_norm
        # layers are performed before the train_step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                _optimiser()
        else:
            _optimiser()
