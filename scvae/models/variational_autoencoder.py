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

import copy
import os
import shutil
from time import time

import numpy
import scipy.sparse
import tensorflow as tf
import tensorflow_probability as tfp

from scvae.data.data_set import DataSet
from scvae.defaults import defaults
from scvae.distributions import (
    DISTRIBUTIONS, LATENT_DISTRIBUTIONS, parse_distribution, Categorised)
from scvae.models.utilities import (
    dense_layer, dense_layers, log_reduce_exp,
    build_training_string, build_data_string,
    early_stopping_status, load_learning_curves,
    generate_unique_run_id_for_model, check_run_id,
    correct_model_checkpoint_path, remove_old_checkpoints,
    copy_model_directory, clear_log_directory,
    parse_numbers_of_samples, validate_model_parameters,
    batch_indices_for_subset)
from scvae.utilities import (
    format_duration, format_time,
    normalise_string, capitalise_string)


class VariationalAutoencoder:
    """Variational auto-encoder class.

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
            latent prior distribution: ``"gaussian"`` or
            ``"unit_variance_gaussian"`` (see :ref:`Training a model`).
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
                 minibatch_normalisation=None,
                 batch_correction=None,
                 number_of_batches=None,
                 number_of_warm_up_epochs=None,
                 log_directory=None,
                 **kwargs):

        # Class setup
        super().__init__()

        self.type = "VAE"

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
        self.latent_distribution_name = latent_distribution
        self.latent_distribution = copy.deepcopy(
            LATENT_DISTRIBUTIONS[latent_distribution]
        )

        parameterise_latent_posterior = kwargs.get(
            "parameterise_latent_posterior")
        if parameterise_latent_posterior is None:
            parameterise_latent_posterior = defaults["models"][
                "parameterise_latent_posterior"]
        self.parameterise_latent_posterior = parameterise_latent_posterior

        number_of_latent_clusters = kwargs.get("number_of_latent_clusters")
        if number_of_latent_clusters is None:
            if "mixture" in latent_distribution:
                number_of_latent_clusters = defaults["models"][
                    "number_of_classes"]
            else:
                number_of_latent_clusters = 1
        self.number_of_latent_clusters = number_of_latent_clusters

        analytical_kl_term = kwargs.get("analytical_kl_term")
        if analytical_kl_term is None:
            if self.latent_distribution_name == "gaussian":
                analytical_kl_term = True
            else:
                analytical_kl_term = False
        self.analytical_kl_term = analytical_kl_term

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

        inference_architecture = kwargs.get("inference_architecture")
        if inference_architecture is None:
            inference_architecture = defaults["models"][
                "inference_architecture"]
        self.inference_architecture = inference_architecture.upper()

        generative_architecture = kwargs.get("generative_architecture")
        if generative_architecture is None:
            generative_architecture = defaults["models"][
                "generative_architecture"]
        self.generative_architecture = generative_architecture.upper()

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

        # Dropout keep probabilities (p) for 3 different kinds of layers
        dropout_keep_probabilities = kwargs.get("dropout_keep_probabilities")
        if dropout_keep_probabilities is None:
            dropout_keep_probabilities = defaults["models"][
                "dropout_keep_probabilities"]
        self.dropout_keep_probabilities = dropout_keep_probabilities
        self.dropout_keep_probability_z = False
        self.dropout_keep_probability_x = False
        self.dropout_keep_probability_h = False
        self.dropout_parts = []
        if isinstance(dropout_keep_probabilities, (list, tuple)):
            p_len = len(dropout_keep_probabilities)
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
            number_of_reconstruction_classes=self.k_max,
            model_type=self.type,
            latent_distribution=self.latent_distribution_name,
            parameterise_latent_posterior=self.parameterise_latent_posterior
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
            warm_up_weight_summary = tf.summary.scalar(
                name="warm_up_weight",
                tensor=self.warm_up_weight
            )
            self.parameter_summary_list.append(warm_up_weight_summary)

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
            self.use_deterministic_z = tf.placeholder(
                dtype=tf.bool,
                shape=[],
                name="use_deterministic_z"
            )

            self.number_of_iw_samples = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="number_of_iw_samples"
            )
            self.number_of_mc_samples = tf.placeholder(
                dtype=tf.int32,
                shape=[],
                name="number_of_mc_samples"
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

        major_parts = [normalise_string(self.latent_distribution_name)]

        if "mixture" in self.latent_distribution_name:
            major_parts.append("c_{}".format(self.number_of_latent_clusters))

        if self.parameterise_latent_posterior:
            major_parts.append("parameterised")

        if self.inference_architecture != "MLP":
            major_parts.append("ia_{}".format(self.inference_architecture))

        if self.generative_architecture != "MLP":
            major_parts.append("ga_{}".format(self.generative_architecture))

        minor_parts = [normalise_string(self.reconstruction_distribution_name)]

        if self.k_max:
            minor_parts.append("k_{}".format(self.k_max))

        if self.use_count_sum_as_feature:
            minor_parts.append("sum")

        minor_parts.append("l_{}".format(self.latent_size))
        minor_parts.append("h_" + "_".join(map(str, self.hidden_sizes)))

        minor_parts.append(
            "mc_{}".format(self.number_of_monte_carlo_samples["training"]))
        minor_parts.append(
            "iw_{}".format(self.number_of_importance_samples["training"]))

        if self.analytical_kl_term:
            minor_parts.append("kl")

        if self.minibatch_normalisation:
            minor_parts.append("bn")

        if self.batch_correction:
            minor_parts.append("bc")

        if len(self.dropout_parts) > 0:
            minor_parts.append("dropout_" + "_".join(self.dropout_parts))

        if self.kl_weight_value != 1:
            minor_parts.append("klw_{}".format(self.kl_weight_value))

        if self.number_of_warm_up_epochs:
            minor_parts.append("wu_{}".format(self.number_of_warm_up_epochs))

        major_part = "-".join(major_parts)
        minor_part = "-".join(minor_parts)

        model_name = os.path.join(self.type, major_part, minor_part)

        return model_name

    @property
    def description(self):
        """Description of model."""

        description_parts = ["Model setup:"]

        description_parts.append("type: {}".format(self.type))
        description_parts.append("feature size: {}".format(self.feature_size))
        description_parts.append("latent size: {}".format(self.latent_size))
        description_parts.append("hidden sizes: {}".format(
            ", ".join(map(str, self.hidden_sizes))))

        description_parts.append(
            "latent distribution: " + self.latent_distribution_name)
        if "mixture" in self.latent_distribution_name:
            description_parts.append("latent clusters: {}".format(
                self.number_of_latent_clusters))
        if self.parameterise_latent_posterior:
            description_parts.append(
                "using parameterisation of latent posterior parameters")

        description_parts.append(
            "reconstruction distribution: "
            + self.reconstruction_distribution_name
        )
        if self.k_max > 0:
            description_parts.append(
                "reconstruction classes: {} (including 0s)".format(self.k_max))

        mc_train = self.number_of_monte_carlo_samples["training"]
        mc_eval = self.number_of_monte_carlo_samples["evaluation"]

        mc = "Monte Carlo samples: {}".format(mc_train)
        if mc_eval != mc_train:
            mc += " (training), {} (evaluation)".format(mc_eval)
        description_parts.append(mc)

        iw_train = self.number_of_importance_samples["training"]
        iw_eval = self.number_of_importance_samples["evaluation"]

        iw = "importance samples: {}".format(iw_train)
        if iw_eval != iw_train:
            iw += " (training), {} (evaluation)".format(iw_eval)
        description_parts.append(iw)

        if self.kl_weight_value != 1:
            description_parts.append(
                "KL weigth: {}".format(self.kl_weight_value))

        if self.analytical_kl_term:
            description_parts.append("using analytical KL term")

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

            if os.path.exists(early_stopping_log_directory):
                validation_losses = load_learning_curves(
                    model=self,
                    data_set_kinds="validation",
                    run_id=run_id,
                    log_directory=log_directory
                )["lower_bound"]
                stopped_early, epochs_with_no_improvement = (
                    early_stopping_status(
                        validation_losses,
                        self.early_stopping_rounds
                    )
                )

        return stopped_early, epochs_with_no_improvement

    def train(self, training_set, validation_set=None, number_of_epochs=None,
              minibatch_size=None, learning_rate=None, run_id=None,
              new_run=None, reset_training=None, **kwargs):
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
            model_string=model_string,
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

        # Adjust minibatch size to account for the number of samples
        minibatch_size /= (
            self.number_of_importance_samples["training"]
            * self.number_of_monte_carlo_samples["training"])
        minibatch_size = int(numpy.ceil(minibatch_size))

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
                "kl_divergence": [],
            }
        }
        if validation_set:
            learning_curves["validation"] = {
                "lower_bound": [],
                "reconstruction_error": [],
                "kl_divergence": [],
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
                    lower_bound_valid_maximum = -numpy.inf
                    epochs_with_no_improvement = 0
                    lower_bound_valid_early_stopping = -numpy.inf
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

                    x_train = noisy_preprocess(training_set.values)
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
                        self.use_deterministic_z: False,
                        self.learning_rate: learning_rate,
                        self.warm_up_weight: warm_up_weight,
                        self.number_of_iw_samples:
                            self.number_of_importance_samples["training"],
                        self.number_of_mc_samples:
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

                # Prior centroids
                p_z_probabilities, p_z_means, p_z_variances = session.run([
                    self.p_z_probabilities,
                    self.p_z_means,
                    self.p_z_variances
                ])

                # Training evaluation
                evaluating_time_start = time()

                lower_bound_train = 0
                kl_divergence_train = 0
                reconstruction_error_train = 0

                q_z_mean_train = numpy.empty(
                    shape=[n_examples_train, self.latent_size],
                    dtype=numpy.float32
                )

                if "mixture" in self.latent_distribution_name:
                    kl_divergence_neurons = numpy.zeros(shape=1)
                else:
                    kl_divergence_neurons = numpy.zeros(shape=self.latent_size)

                for i in range(0, n_examples_train, minibatch_size):
                    subset = slice(
                        i, min(i + minibatch_size, n_examples_train))
                    x_batch = x_train[subset].toarray()
                    t_batch = t_train[subset].toarray()
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.use_deterministic_z: False,
                        self.warm_up_weight: 1.0,
                        self.number_of_iw_samples:
                            self.number_of_importance_samples["training"],
                        self.number_of_mc_samples:
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
                        lower_bound_i,
                        kl_divergence_i,
                        reconstruction_error_i,
                        q_z_mean_i,
                        kl_divergence_neurons_i
                    ) = session.run(
                        [
                            self.lower_bound,
                            self.kl_divergence,
                            self.reconstruction_error,
                            self.q_z_mean,
                            self.kl_divergence_neurons
                        ],
                        feed_dict=feed_dict_batch
                    )

                    lower_bound_train += lower_bound_i
                    kl_divergence_train += kl_divergence_i
                    reconstruction_error_train += reconstruction_error_i

                    q_z_mean_train[subset] = q_z_mean_i

                    kl_divergence_neurons += kl_divergence_neurons_i

                lower_bound_train /= n_examples_train / minibatch_size
                kl_divergence_train /= n_examples_train / minibatch_size
                reconstruction_error_train /= n_examples_train / minibatch_size

                kl_divergence_neurons /= n_examples_train / minibatch_size

                learning_curves["training"]["lower_bound"].append(
                    lower_bound_train)
                learning_curves["training"]["reconstruction_error"].append(
                    reconstruction_error_train)
                learning_curves["training"]["kl_divergence"].append(
                    kl_divergence_train)

                evaluating_duration = time() - evaluating_time_start

                # Training summaries
                training_summary = tf.Summary()

                # Training loss summaries
                training_summary.value.add(
                    tag="losses/lower_bound",
                    simple_value=lower_bound_train
                )
                training_summary.value.add(
                    tag="losses/reconstruction_error",
                    simple_value=reconstruction_error_train
                )
                training_summary.value.add(
                    tag="losses/kl_divergence",
                    simple_value=kl_divergence_train
                )

                # Training KL divergence summaries
                for i in range(kl_divergence_neurons.size):
                    training_summary.value.add(
                        tag="kl_divergence_neurons/{}".format(i),
                        simple_value=kl_divergence_neurons[i]
                    )

                # Training centroid summaries
                if not validation_set:
                    for k in range(len(p_z_probabilities)):
                        training_summary.value.add(
                            tag="prior/cluster_{}/probability".format(k),
                            simple_value=p_z_probabilities[k]
                        )
                        for l in range(self.latent_size):
                            # The same Gaussian for all
                            if not p_z_means[k].shape:
                                p_z_mean_k_l = p_z_means[k]
                                p_z_variances_k_l = p_z_variances[k]
                            # Different Gaussians for all
                            else:
                                p_z_mean_k_l = p_z_means[k][l]
                                p_z_variances_k_l = p_z_variances[k][l]
                            training_summary.value.add(
                                tag="prior/cluster_{}/mean/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_mean_k_l
                            )
                            training_summary.value.add(
                                tag="prior/cluster_{}/variance/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_variances_k_l
                            )

                # Writing training summaries
                training_summary_writer.add_summary(
                    training_summary,
                    global_step=epoch + 1
                )
                training_summary_writer.flush()

                # Printing training evaluation
                print(
                    "    {} set ({}):".format(
                        training_set.kind.capitalize(),
                        format_duration(evaluating_duration)
                    ),
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                        lower_bound_train,
                        reconstruction_error_train,
                        kl_divergence_train
                    )
                )

                if validation_set:

                    # Validation evaluation
                    evaluating_time_start = time()

                    lower_bound_valid = 0
                    kl_divergence_valid = 0
                    reconstruction_error_valid = 0

                    q_z_mean_valid = numpy.empty(
                        shape=[n_examples_valid, self.latent_size],
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
                            self.use_deterministic_z: False,
                            self.warm_up_weight: 1.0,
                            self.number_of_iw_samples:
                                self.number_of_importance_samples["training"],
                            self.number_of_mc_samples:
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
                            lower_bound_i,
                            kl_divergence_i,
                            reconstruction_error_i,
                            q_z_mean_i
                        ) = session.run(
                            [
                                self.lower_bound,
                                self.kl_divergence,
                                self.reconstruction_error,
                                self.q_z_mean
                            ],
                            feed_dict=feed_dict_batch
                        )

                        lower_bound_valid += lower_bound_i
                        kl_divergence_valid += kl_divergence_i
                        reconstruction_error_valid += reconstruction_error_i

                        q_z_mean_valid[subset] = q_z_mean_i

                    lower_bound_valid /= n_examples_valid / minibatch_size
                    kl_divergence_valid /= n_examples_valid / minibatch_size
                    reconstruction_error_valid /= (
                        n_examples_valid / minibatch_size)

                    learning_curves["validation"]["lower_bound"].append(
                        lower_bound_valid)
                    learning_curves["validation"][
                         "reconstruction_error"].append(
                             reconstruction_error_valid)
                    learning_curves["validation"]["kl_divergence"].append(
                        kl_divergence_valid)

                    evaluating_duration = time() - evaluating_time_start

                    # Validation summaries
                    summary = tf.Summary()

                    # Validation loss summaries
                    summary.value.add(
                        tag="losses/lower_bound",
                        simple_value=lower_bound_valid
                    )
                    summary.value.add(
                        tag="losses/reconstruction_error",
                        simple_value=reconstruction_error_valid
                    )
                    summary.value.add(
                        tag="losses/kl_divergence",
                        simple_value=kl_divergence_valid
                    )

                    # Validation centroid summaries
                    for k in range(len(p_z_probabilities)):
                        summary.value.add(
                            tag="prior/cluster_{}/probability".format(k),
                            simple_value=p_z_probabilities[k]
                        )
                        for l in range(self.latent_size):
                            # The same Gaussian for all
                            if not p_z_means[k].shape:
                                p_z_mean_k_l = p_z_means[k]
                                p_z_variances_k_l = p_z_variances[k]
                            # Different Gaussians for all
                            else:
                                p_z_mean_k_l = p_z_means[k][l]
                                p_z_variances_k_l = p_z_variances[k][l]
                            summary.value.add(
                                tag="prior/cluster_{}/mean/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_mean_k_l
                            )
                            summary.value.add(
                                tag="prior/cluster_{}/variance/dimension_{}"
                                    .format(k, l),
                                simple_value=p_z_variances_k_l
                            )

                    # Writing validation summaries
                    validation_summary_writer.add_summary(
                        summary,
                        global_step=epoch + 1
                    )
                    validation_summary_writer.flush()

                    # Printing validation summaries
                    print(
                        "    {} set ({}):".format(
                            validation_set.kind.capitalize(),
                            format_duration(evaluating_duration)
                        ),
                        "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                            lower_bound_valid,
                            reconstruction_error_valid,
                            kl_divergence_valid
                        )
                    )

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
                                    epochs_with_no_improvement + 1
                                )
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
                    session,
                    checkpoint_file,
                    global_step=epoch + 1
                )
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
                            n_clusters = len(p_z_probabilities)
                            n_latent = self.latent_size
                            p_z_covariance_matrices = numpy.empty(
                                shape=[n_clusters, n_latent, n_latent])
                            for k in range(n_clusters):
                                p_z_covariance_matrices[k] = numpy.diag(
                                    p_z_variances[k])
                            centroids = {
                                "prior": {
                                    "probabilities": numpy.array(
                                        p_z_probabilities),
                                    "means": numpy.stack(p_z_means),
                                    "covariance_matrices": (
                                        p_z_covariance_matrices)
                                }
                            }
                        else:
                            centroids = None

                        if validation_set:
                            intermediate_latent_values = q_z_mean_valid
                            intermediate_data_set = validation_set
                        else:
                            intermediate_latent_values = q_z_mean_train
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

            z_samples = numpy.empty(
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

                z_samples_i = session.run(
                    self.p_z_samples,
                    feed_dict={self.sample_size: minibatch_sample_size}
                )
                z_samples[indices] = z_samples_i

                feed_dict_batch = {
                    self.z: z_samples[indices],
                    self.is_training: False,
                    self.number_of_iw_samples: 1,
                    self.number_of_mc_samples: 1
                }

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
                values=z_samples,
                preprocessed_values=None,
                labels=None,
                example_names=sample_names,
                feature_names=numpy.array(["latent variable {}".format(
                    i + 1) for i in range(self.latent_size)]),
                batch_indices=batch_indices_samples,
                feature_selection=None,
                example_filter=None,
                preprocessing_methods=None,
                kind="sample",
                version="z"
            )

            sample_latent_sets = {
                "z": sample_z_set
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
            data set of the latent variable (wrapped in a dictionary
            for compatibility with
            `GaussianMixtureVariationalAutoencoder`).
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
                    "Can only output at most 3 sets, {} requested".format(
                        number_of_output_versions)
                )
            elif number_of_output_versions != len(set(output_versions)):
                raise ValueError(
                    "Cannot output duplicate sets, {} requested.".format(
                        output_versions)
                )

        evaluation_subset_indices = kwargs.get("evaluation_subset_indices")
        if evaluation_subset_indices is None:
            evaluation_subset_indices = set()

        evaluation_set_transformed = False

        minibatch_size /= (
            self.number_of_importance_samples["evaluation"]
            * self.number_of_monte_carlo_samples["evaluation"]
        )
        minibatch_size = int(numpy.ceil(minibatch_size))

        if self.batch_correction:
            batch_indices_eval = batch_indices_for_subset(evaluation_set)

        if self.use_count_sum_as_parameter:
            count_sum_parameter_eval = evaluation_set.count_sum

        if self.use_count_sum_as_feature:
            count_sum_feature_eval = evaluation_set.normalised_count_sum

        n_examples_eval = evaluation_set.number_of_examples
        n_features_eval = evaluation_set.number_of_features

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

        # max_count = int(max(t_eval, axis = (0, 1)))

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
            kl_divergence_eval = 0
            reconstruction_error_eval = 0

            if log_results:
                if "mixture" in self.latent_distribution_name:
                    kl_divergence_neurons = numpy.zeros(shape=1)
                else:
                    kl_divergence_neurons = numpy.zeros(shape=self.latent_size)

            if "reconstructed" in output_versions:
                p_x_mean_eval = numpy.empty(
                    shape=(n_examples_eval, n_features_eval),
                    dtype=numpy.float32
                )
                p_x_stddev_eval = scipy.sparse.lil_matrix(
                    (n_examples_eval, n_features_eval),
                    dtype=numpy.float32
                )
                stddev_of_p_x_mean_eval = scipy.sparse.lil_matrix(
                    (n_examples_eval, n_features_eval),
                    dtype=numpy.float32
                )

            if "latent" in output_versions:
                q_z_mean_eval = numpy.empty(
                    shape=(n_examples_eval, self.latent_size),
                    dtype=numpy.float32
                )

            use_deterministic_z = kwargs.get("use_deterministic_z", False)
            if use_deterministic_z:
                number_of_iw_samples = 1
                number_of_mc_samples = 1
            else:
                number_of_iw_samples = self.number_of_importance_samples[
                    "evaluation"]
                number_of_mc_samples = self.number_of_monte_carlo_samples[
                    "evaluation"]

            for i in range(0, n_examples_eval, minibatch_size):

                indices = numpy.arange(
                    i, min(i + minibatch_size, n_examples_eval))

                subset_indices = numpy.array(list(
                    evaluation_subset_indices.intersection(indices)))

                feed_dict_batch = {
                    self.x: x_eval[indices].toarray(),
                    self.t: t_eval[indices].toarray(),
                    self.is_training: False,
                    self.use_deterministic_z: use_deterministic_z,
                    self.warm_up_weight: 1.0,
                    self.number_of_iw_samples: number_of_iw_samples,
                    self.number_of_mc_samples: number_of_mc_samples
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
                    lower_bound_i,
                    kl_divergence_i,
                    reconstruction_error_i,
                    p_x_mean_i, p_x_stddev_i, stddev_of_p_x_mean_i,
                    q_z_mean_i, kl_divergence_neurons_i
                ) = session.run(
                    [
                        self.lower_bound,
                        self.kl_divergence,
                        self.reconstruction_error, self.p_x_mean,
                        self.p_x_stddev, self.stddev_of_p_x_given_z_mean,
                        self.q_z_mean, self.kl_divergence_neurons
                    ],
                    feed_dict=feed_dict_batch
                )

                lower_bound_eval += lower_bound_i
                kl_divergence_eval += kl_divergence_i
                reconstruction_error_eval += reconstruction_error_i

                if log_results:
                    kl_divergence_neurons += numpy.array(
                        kl_divergence_neurons_i)

                if "reconstructed" in output_versions:
                    # Save Importance weighted Monte Carlo estimates of:
                    # Reconstruction mean (marginalised conditional mean):
                    #      E[x] = E[E[x|z]] = E_q(z|x)[E_p(x|z)[x]]
                    #           = E_z[p_x_given_z.mean]
                    #     \approx 1/(R*L) \sum^R_r w_r \sum^L_{l=1}
                    # p_x_given_z.mean
                    p_x_mean_eval[indices] = p_x_mean_i

                    if subset_indices.size > 0:

                        # Reconstruction standard deviation:
                        #     sqrt(V[x]) = sqrt(E[V[x|z]] + V[E[x|z]])
                        #     = E_z[p_x_given_z.var] + E_z[(p_x_given_z.mean
                        #       - E[x])^2]
                        p_x_stddev_eval[subset_indices] = p_x_stddev_i[
                            subset_indices - i]

                        # Estimated standard deviation of Monte Carlo estimate
                        # E[x].
                        stddev_of_p_x_mean_eval[subset_indices] = (
                            stddev_of_p_x_mean_i[subset_indices - i])

                if "latent" in output_versions:
                    q_z_mean_eval[indices] = q_z_mean_i

            lower_bound_eval /= n_examples_eval / minibatch_size
            kl_divergence_eval /= n_examples_eval / minibatch_size
            reconstruction_error_eval /= n_examples_eval / minibatch_size

            if log_results:
                kl_divergence_neurons /= n_examples_eval / minibatch_size

            evaluating_duration = time() - evaluating_time_start

            # Summaries

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
                    tag="losses/kl_divergence",
                    simple_value=kl_divergence_eval
                )

                # Centroid summaries

                p_z_probabilities, p_z_means, p_z_variances = session.run([
                    self.p_z_probabilities, self.p_z_means, self.p_z_variances
                ])

                for k in range(len(p_z_probabilities)):
                    summary.value.add(
                        tag="prior/cluster_{}/probability".format(k),
                        simple_value=p_z_probabilities[k]
                    )
                    for l in range(self.latent_size):
                        # The same Gaussian for all
                        if not p_z_means[k].shape:
                            p_z_mean_k_l = p_z_means[k]
                            p_z_variances_k_l = p_z_variances[k]
                        # Different Gaussians for all
                        else:
                            p_z_mean_k_l = p_z_means[k][l]
                            p_z_variances_k_l = p_z_variances[k][l]
                        summary.value.add(
                            tag="prior/cluster_{}/mean/dimension_{}".format(
                                k, l),
                            simple_value=p_z_mean_k_l
                        )
                        summary.value.add(
                            tag="prior/cluster_{}/variance/dimension_{}"
                                .format(k, l),
                            simple_value=p_z_variances_k_l
                        )

                for l in range(kl_divergence_neurons.size):
                    summary.value.add(
                        tag="kl_divergence_neurons/{}".format(l),
                        simple_value=kl_divergence_neurons[l]
                    )

                # Write summaries
                eval_summary_writer.add_summary(summary, global_step=epoch)
                eval_summary_writer.flush()

            # Print evaluation
            print(
                "    {} set ({}): ".format(
                    evaluation_set.kind.capitalize(),
                    format_duration(evaluating_duration)
                ),
                "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                    lower_bound_eval,
                    reconstruction_error_eval,
                    kl_divergence_eval
                )
            )

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
                    explained_standard_deviations=stddev_of_p_x_mean_eval,
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
                    values=q_z_mean_eval,
                    preprocessed_values=None,
                    labels=evaluation_set.labels,
                    example_names=evaluation_set.example_names,
                    feature_names=numpy.array(["latent variable {}".format(
                        i + 1) for i in range(self.latent_size)]),
                    batch_indices=evaluation_set.batch_indices,
                    batch_names=evaluation_set.batch_names,
                    features_mapped=evaluation_set.features_mapped,
                    feature_selection=evaluation_set.feature_selection,
                    example_filter=evaluation_set.example_filter,
                    preprocessing_methods=evaluation_set.preprocessing_methods,
                    kind=evaluation_set.kind,
                    version="z"
                )

                latent_evaluation_sets = {
                    "z": z_evaluation_set
                }

                index = output_versions.index("latent")
                output_sets[index] = latent_evaluation_sets

            if len(output_sets) == 1:
                output_sets = output_sets[0]

            return output_sets

    def _setup_model_graph(self):

        if self.inference_architecture == "MLP":
            encoder = dense_layers(
                inputs=self.x,
                num_outputs=self.hidden_sizes,
                activation_fn=tf.nn.relu,
                minibatch_normalisation=self.minibatch_normalisation,
                is_training=self.is_training,
                input_dropout_keep_probability=self.dropout_keep_probability_x,
                hidden_dropout_keep_probability=(
                    self.dropout_keep_probability_h),
                scope="ENCODER"
            )
        elif self.inference_architecture == "LFM":
            encoder = self.x
        else:
            raise ValueError(
                "The generative architecture can only be a neural network "
                "(MLP) or a linear factor model (LFM)."
            )

        # Parameterising the approximate posterior and prior over z

        for part_name in self.latent_distribution:
            part_distribution_name = self.latent_distribution[part_name][
                "name"]
            part_distribution = DISTRIBUTIONS[part_distribution_name]
            part_parameters = self.latent_distribution[part_name]["parameters"]

            with tf.variable_scope(part_name.upper()):

                # Retrieving layers for all latent distribution parameters
                for parameter_name in part_distribution["parameters"]:
                    if parameter_name in part_parameters:
                        part_parameters[parameter_name] = tf.expand_dims(
                            tf.expand_dims(
                                tf.constant(
                                    part_parameters[parameter_name],
                                    dtype=tf.float32
                                ),
                                axis=0
                            ),
                            axis=0
                        )
                        continue

                    parameter_activation_function = part_distribution[
                        "parameters"][parameter_name]["activation function"]
                    p_min, p_max = part_distribution["parameters"][
                        parameter_name]["support"]
                    initial_value = part_distribution["parameters"][
                        parameter_name]["initial value"]

                    def parameter_variable(num_outputs, name):
                        def activation_function(x):
                            return tf.clip_by_value(
                                parameter_activation_function(x),
                                p_min + numpy.finfo(dtype=numpy.float32).tiny,
                                p_max - numpy.finfo(dtype=numpy.float32).tiny
                            )
                        if part_name == "posterior":
                            variable = dense_layer(
                                inputs=encoder,
                                num_outputs=num_outputs,
                                activation_fn=activation_function,
                                is_training=self.is_training,
                                dropout_keep_probability=(
                                    self.dropout_keep_probability_h),
                                scope=name
                            )
                        elif part_name == "prior":
                            variable = tf.expand_dims(
                                tf.Variable(
                                    activation_function(
                                        initial_value([num_outputs])
                                    ),
                                    name=name
                                ),
                                axis=0
                            )
                        return variable

                    if "mixture" in part_distribution_name:
                        if parameter_name == "logits":
                            logits = tf.expand_dims(tf.expand_dims(
                                parameter_variable(
                                    num_outputs=self.number_of_latent_clusters,
                                    name=parameter_name.upper()
                                ), axis=0), axis=0)
                            part_parameters[parameter_name] = logits
                        else:
                            part_parameters[parameter_name] = []
                            for k in range(self.number_of_latent_clusters):
                                part_parameters[parameter_name].append(
                                    tf.expand_dims(tf.expand_dims(
                                        parameter_variable(
                                            num_outputs=self.latent_size,
                                            name=(parameter_name[:-1].upper()
                                                  + "_" + str(k))
                                        ), axis=0), axis=0
                                    )
                                )
                    else:
                        part_parameters[parameter_name] = tf.expand_dims(
                            tf.expand_dims(
                                parameter_variable(
                                    num_outputs=self.latent_size,
                                    name=parameter_name.upper()
                                ), axis=0
                            ), axis=0
                        )

        # Latent posterior distribution
        if self.parameterise_latent_posterior:
            posterior_parameters = self.latent_distribution["posterior"][
                "parameters"]
            prior_parameters = self.latent_distribution["prior"]["parameters"]
            for parameter in posterior_parameters:
                if isinstance(posterior_parameters[parameter], list):
                    for k in range(self.number_of_latent_clusters):
                        posterior_parameters[parameter][k] += prior_parameters[
                            parameter][k]
                else:
                    posterior_parameters[parameter] += prior_parameters[
                        parameter]

        self.q_z_given_x = DISTRIBUTIONS[self.latent_distribution["posterior"][
            "name"]]["class"](self.latent_distribution["posterior"][
                "parameters"])

        self.q_z_mean = self.q_z_given_x.mean()

        # Sampling of importance weighting and Monte Carlo samples
        total_number_of_samples = tf.where(
            self.use_deterministic_z,
            1,
            self.number_of_iw_samples * self.number_of_mc_samples
        )
        self.z = tf.cast(
            tf.reshape(
                tf.cond(
                    self.use_deterministic_z,
                    lambda: tf.expand_dims(self.q_z_mean, axis=0),
                    lambda: self.q_z_given_x.sample(total_number_of_samples)
                ),
                shape=[-1, self.latent_size],
                name="Z"
            ),
            dtype=tf.float32
        )

        # Latent prior distribution
        self.p_z = DISTRIBUTIONS[self.latent_distribution["prior"]["name"]][
            "class"](self.latent_distribution["prior"]["parameters"])

        self.p_z_probabilities = []
        self.p_z_means = []
        self.p_z_variances = []

        if "mixture" in self.latent_distribution["prior"]["name"]:
            for k in range(self.number_of_latent_clusters):
                self.p_z_probabilities.append(
                    tf.squeeze(self.p_z.cat.probs)[k])
                self.p_z_means.append(
                    tf.squeeze(self.p_z.components[k].mean()))
                self.p_z_variances.append(
                    tf.diag_part(tf.squeeze(
                        self.p_z.components[k].covariance())))
        else:
            self.p_z_probabilities.append(tf.constant(1.))
            self.p_z_means.append(tf.squeeze(self.p_z.mean()))
            self.p_z_variances.append(tf.squeeze(self.p_z.stddev()))

        self.p_z_samples = tf.squeeze(
            self.p_z.sample(sample_shape=(self.sample_size, self.latent_size)))

        # Decoder - Generative model, p(x|z)

        # Make sure we use a replication per sample of the feature sum,
        # when adding this to the features.
        if self.use_count_sum_as_parameter:
            replicated_count_sum_parameter = tf.tile(
                self.count_sum_parameter,
                multiples=[
                    self.number_of_iw_samples * self.number_of_mc_samples, 1]
            )

        decoder_inputs = [self.z]

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
                multiples=[
                    self.number_of_iw_samples * self.number_of_mc_samples, 1],
                name="BATCH_INDICES"
            )
            decoder_inputs.append(replicated_batch_indices)

        if self.use_count_sum_as_feature:
            replicated_count_sum_feature = tf.tile(
                self.count_sum_feature,
                multiples=[
                    self.number_of_iw_samples * self.number_of_mc_samples, 1],
                name="COUNT_SUM"
            )
            decoder_inputs.append(replicated_count_sum_feature)

        if len(decoder_inputs) > 1:
            decoder = tf.concat(
                decoder_inputs,
                axis=1,
                name="-".join(i.name.replace(":0", "") for i in decoder_inputs)
            )
        else:
            decoder = self.z

        if self.generative_architecture == "MLP":
            decoder = dense_layers(
                inputs=decoder,
                num_outputs=self.hidden_sizes,
                reverse_order=True,
                activation_fn=tf.nn.relu,
                minibatch_normalisation=self.minibatch_normalisation,
                is_training=self.is_training,
                input_dropout_keep_probability=self.dropout_keep_probability_z,
                hidden_dropout_keep_probability=(
                    self.dropout_keep_probability_h),
                scope="DECODER"
            )
        elif self.generative_architecture == "LFM":
            pass
        else:
            raise ValueError(
                "The inference architecture can only be a neural network "
                "(MLP) or a linear factor model (LFM)."
            )

        # Reconstruction distribution parameterisation

        with tf.variable_scope("X_TILDE"):

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
                    scope=parameter.upper()
                )

            if ("constrained" in self.reconstruction_distribution_name
                    or "multinomial" in self.reconstruction_distribution_name):
                self.p_x_given_z = self.reconstruction_distribution["class"](
                    theta=x_theta,
                    N=replicated_count_sum_parameter
                )
            elif "multinomial" in self.reconstruction_distribution_name:
                self.p_x_given_z = self.reconstruction_distribution["class"](
                    theta=x_theta,
                    N=replicated_count_sum_parameter
                )
            else:
                self.p_x_given_z = self.reconstruction_distribution["class"](
                    theta=x_theta
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
                    scope="P_K"
                )

                x_logits = tf.reshape(
                    x_logits,
                    shape=[
                        -1,
                        self.feature_size,
                        self.number_of_reconstruction_classes
                    ]
                )

                self.p_x_given_z = Categorised(
                    dist=self.p_x_given_z,
                    cat=tfp.distributions.Categorical(logits=x_logits)
                )

            self.p_x_given_z_mean = tf.reshape(
                self.p_x_given_z.mean(),
                shape=[
                    self.number_of_iw_samples,
                    self.number_of_mc_samples,
                    -1,
                    self.feature_size
                ]
            )

            self.p_x_given_z_variance = tf.reshape(
                self.p_x_given_z.variance(),
                shape=[
                    self.number_of_iw_samples,
                    self.number_of_mc_samples,
                    -1,
                    self.feature_size
                ]
            )

        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            parameter_summary = tf.summary.histogram(parameter.name, parameter)
            self.parameter_summary_list.append(parameter_summary)
        self.parameter_summary = tf.summary.merge(self.parameter_summary_list)

    def _setup_loss_function(self):

        # Prepare replicated and reshaped arrays by replicating out
        # minibatches in tiles per sample into a shape of (R * L * B, D_x)
        t_tiled = tf.tile(
            self.t,
            multiples=[self.number_of_iw_samples*self.number_of_mc_samples, 1])
        # Reshape samples back to (R, L, B, D_z)
        z_reshaped = tf.reshape(
            self.z,
            shape=[
                self.number_of_iw_samples,
                self.number_of_mc_samples,
                -1,
                self.latent_size
            ]
        )

        # Reconstruction error
        # 1. Evaluate all log(p(x|z)) (R * L * B, D_x) target values
        #    in the (R * L * B, D_x) probability distributions learned
        # 2. Sum over all N_x features
        # 3. and reshape it back to (R, L, B)
        p_x_given_z_log_prob = self.p_x_given_z.log_prob(t_tiled)
        log_p_x_given_z = tf.reshape(
            tf.reduce_sum(
                p_x_given_z_log_prob,
                axis=-1
            ),
            shape=[self.number_of_iw_samples, self.number_of_mc_samples, -1]
        )

        # Average over all samples and examples and add to losses in summary
        self.reconstruction_error = tf.reduce_mean(log_p_x_given_z)
        tf.add_to_collection(name="losses", value=self.reconstruction_error)

        # Recognition error
        if "mixture" in self.latent_distribution_name:
            # Evaluate Kullback-Leibler divergence numerically with sampling
            # Evaluate all log(q(z|x)) and log(p(z)) on (R, L, B, D_z)
            # latent sample values.
            # Shape: (R, L, B, D_z)
            log_q_z_given_x = self.q_z_given_x.log_prob(z_reshaped)
            # Shape:  (R, L, B, D_z)
            log_p_z = self.p_z.log_prob(z_reshaped)

            if "mixture" not in self.latent_distribution["posterior"]["name"]:
                log_q_z_given_x = tf.reduce_sum(log_q_z_given_x, axis=-1)
            if "mixture" not in self.latent_distribution["prior"]["name"]:
                log_p_z = tf.reduce_sum(log_p_z, axis=-1)

            # Kullback-Leibler divergence:
            # KL[q(z|x)||p(z)] = E_q[log(q(z|x)/p(z))]
            #                  = E_q[log(q(z|x))] - E_q[log(p(z))]
            kl_divergence = log_q_z_given_x - log_p_z

            # KL regularisation term
            self.kl_divergence = tf.reduce_mean(
                kl_divergence, name="kl_divergence")
            tf.add_to_collection("losses", value=self.kl_divergence)
            # Mean KL for all D_z dimensions --> shape = (1)
            self.kl_divergence_neurons = tf.expand_dims(
                tf.reduce_mean(kl_divergence), axis=-1)
        else:
            if self.analytical_kl_term:
                # Evaluate KL divergence analytically without sampling
                kl_divergence = tfp.distributions.kl_divergence(
                    self.q_z_given_x, self.p_z)
            else:
                # Evaluate KL divergence numerically with sampling
                # Evaluate all log(q(z|x)) and log(p(z)) on (R, L, B,
                # D_z) latent sample values.
                # Shape: (R, L, B, D_z)
                log_q_z_given_x = self.q_z_given_x.log_prob(z_reshaped)
                # Shape: (R, L, B, D_z)
                log_p_z = self.p_z.log_prob(z_reshaped)

                # Kullback-Leibler divergence:
                # KL[q(z|x)||p(z)] = E_q[log(q(z|x)/p(z))]
                #                  = E_q[log(q(z|x))] - E_q[log(p(z))]
                kl_divergence = log_q_z_given_x - log_p_z

            # Mean KL for all D_z dim. --> shape = (D_z)
            self.kl_divergence_neurons = tf.reduce_mean(
                tf.reshape(kl_divergence, shape=[-1, self.latent_size]),
                axis=0
            )
            # KL regularisation term: Sum up KL over all latent dimensions:
            # Shape: ()
            self.kl_divergence = tf.reduce_sum(
                self.kl_divergence_neurons,
                name="kl_divergence"
            )
            tf.add_to_collection("losses", value=self.kl_divergence)

            # Shape: (R, L, B, D_x) --> (R, L, B)
            kl_divergence = tf.reduce_sum(kl_divergence, axis=-1)

            # Importance weighted Monte Carlo estimates of:
            # Reconstruction mean (marginalised conditional mean):
            #      E[x] = E[E[x|z]] = E_q(z|x)[E_p(x|z)[x]]
            #           = E_z[p_x_given_z.mean] = E_z[<x|z_{r,l}>]
            #     \approx Ê[x] = 1/(R*L) \sum^R_r w_r \sum^L_{l=1} <x|z_{r,l}>
            #          where <x|z_{r,l}> is explicitly known from p(x|z).
            # Shape: (R, L, B, D_x) --> (R, B, D_x) --> (B, D_x)
            self.p_x_mean = tf.reduce_mean(
                tf.reduce_mean(self.p_x_given_z_mean, axis=1),
                axis=0
            )

            # Estimated variance of likelihood expectation:
            # ^V[E[x|z]] = ( E[x|z_l] - Ê[x] )^2
            self.variance_of_p_x_given_z_mean = tf.reduce_mean(
                tf.reduce_mean(
                    tf.square(
                        self.p_x_given_z_mean - tf.reshape(
                            self.p_x_mean,
                            shape=[1, 1, -1, self.feature_size]
                        )
                    ),
                    axis=1
                ),
                axis=0
            )

            self.stddev_of_p_x_given_z_mean = tf.sqrt(
                self.variance_of_p_x_given_z_mean)

            # Reconstruction standard deviation of Monte Carlo estimator:
            #      ^V[Ê[x]] = 1/(R*L) * V[E_p(x|z)[x]] = 1/(R*L)^2 * V[<x|z>]
            #              = 1/(R*L) * Ê_z[(E[x|z] - E[x])^2]
            #              = 1/(R*L) * Ê_z[(p_x_given_z.mean - Ê[x])^2]
            #              = 1/(R*L) * ^V[E[x|z]]
            # Shape: (R, L, B, D_x)
            self.variance_of_p_x_mean = (
                self.variance_of_p_x_given_z_mean
                / tf.cast(
                    self.number_of_iw_samples * self.number_of_mc_samples,
                    dtype=tf.float32
                )
            )

            self.stddev_of_p_x_mean = tf.sqrt(self.variance_of_p_x_mean)

            # Estimator for variance decomposition:
            # V[x] = E[V(x|z)] + V[E(x|z)] \approx ^V_z[E[x|z]] + Ê_z[V[x|z]]
            self.p_x_variance = (
                self.variance_of_p_x_given_z_mean + tf.reduce_mean(
                    tf.reduce_mean(self.p_x_given_z_variance, axis=1),
                    axis=0
                )
            )

            self.p_x_stddev = tf.sqrt(self.p_x_variance)

        # average over eq_samples, minibatch_size dimensions    -> shape: ()
        # log-mean-exp (to avoid over- and underflow) over iw_samples dimension
        self.lower_bound = tf.reduce_mean(
            log_reduce_exp(
                log_p_x_given_z - kl_divergence,
                reduction_function=tf.reduce_mean,
                axis=0
            )
        )
        tf.add_to_collection("losses", self.lower_bound)
        self.lower_bound_weighted = tf.reduce_mean(
            log_reduce_exp(
                (
                    log_p_x_given_z
                    - self.warm_up_weight * self.kl_weight * kl_divergence
                ),
                reduction_function=tf.reduce_mean,
                axis=0
            )
        )

    def _setup_optimiser(self):

        # Create the gradient descent optimiser with the given learning rate.
        def _optimiser():

            # Optimiser and training objective of negative loss
            optimiser = tf.train.AdamOptimizer(self.learning_rate)

            # Create a variable to track the global step
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
        # layers are performed before the train_step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                _optimiser()
        else:
            _optimiser()
