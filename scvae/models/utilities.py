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

import os
import random
import re
import shutil
import time
from collections import namedtuple
from datetime import datetime
from string import ascii_uppercase

import numpy
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout

from scvae.utilities import (
    capitalise_string, enumerate_strings, normalise_string)


# Wrapper layer for inserting batch normalisation in between linear and
# nonlinear activation layers
def dense_layer(inputs, num_outputs, is_training=True, scope="layer",
                activation_fn=None, minibatch_normalisation=False, decay=0.999,
                center=True, scale=False, reuse=False,
                dropout_keep_probability=False):

    with tf.variable_scope(scope):
        # Dropout input connections with rate = (1- dropout_keep_probability)
        if dropout_keep_probability and dropout_keep_probability != 1:
            inputs = dropout(
                inputs=inputs,
                keep_prob=dropout_keep_probability,
                is_training=is_training
            )

        # Set up weights for and transform inputs through neural network
        outputs = fully_connected(
            inputs=inputs,
            num_outputs=num_outputs,
            activation_fn=None,
            scope="DENSE",
            reuse=reuse
        )

        # Set up normalisation across examples with learned center and scale
        if minibatch_normalisation:
            outputs = batch_norm(
                inputs=outputs,
                center=center,
                scale=scale,
                is_training=is_training,
                scope="BATCH_NORM",
                reuse=reuse
            )

        # Apply non-linear activation function to linear outputs
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs


# Wrapper layer for inserting batch normalisation in between several linear
# and non-linear activation layers in given or reverse order
def dense_layers(inputs, num_outputs, reverse_order=False, is_training=True,
                 scope="layers", layer_name=None, activation_fn=None,
                 minibatch_normalisation=False, decay=0.999, center=True,
                 scale=False, reuse=False,
                 input_dropout_keep_probability=False,
                 hidden_dropout_keep_probability=False):

    if not isinstance(num_outputs, (list, tuple)):
        num_outputs = [num_outputs]
    if reverse_order:
        num_outputs = num_outputs[::-1]

    if layer_name:
        layer_name = layer_name.upper() + "_"
    else:
        layer_name = ""

    outputs = inputs

    with tf.variable_scope(scope):
        for i, num_output in enumerate(num_outputs):
            if not reverse_order:
                layer_number = i + 1
            else:
                layer_number = len(num_outputs) - i

            if i == 0:
                dropout_keep_probability = input_dropout_keep_probability
            else:
                dropout_keep_probability = hidden_dropout_keep_probability

            outputs = dense_layer(
                inputs=outputs,
                num_outputs=num_output,
                is_training=is_training,
                scope=layer_name + "{:d}".format(layer_number),
                activation_fn=activation_fn,
                minibatch_normalisation=minibatch_normalisation,
                decay=decay,
                center=center,
                scale=scale,
                reuse=reuse,
                dropout_keep_probability=dropout_keep_probability
            )

    return outputs


def log_reduce_exp(input_tensor, reduction_function=tf.reduce_mean, axis=None):
    # log-mean-exp over axis to avoid overflow and underflow
    input_tensor_max = tf.reduce_max(input_tensor, axis=axis, keepdims=True)
    output_tensor = tf.log(reduction_function(
        tf.exp(input_tensor - input_tensor_max),
        axis=axis,
        keepdims=True
    )) + input_tensor_max
    return tf.squeeze(output_tensor)


def build_training_string(model_string, epoch_start, number_of_epochs,
                          data_string):

    if epoch_start == 0:
        training_string = "Training {} for {} epochs on {}.".format(
            model_string, number_of_epochs, data_string)
    elif epoch_start < number_of_epochs:
        training_string = (
            "Continue training {} for {} additionally epochs (up to {} epochs)"
            " on {}.".format(
                model_string,
                number_of_epochs - epoch_start,
                number_of_epochs,
                data_string
            )
        )
    elif epoch_start == number_of_epochs:
        training_string = (
            "{} has already been trained for {} epochs on {}.".format(
                capitalise_string(model_string), number_of_epochs, data_string
            )
        )
    elif epoch_start > number_of_epochs:
        training_string = (
            "{} has already been trained for more than {} epochs on {}. "
            "Loading model trained for {} epochs.".format(
                capitalise_string(model_string),
                number_of_epochs,
                data_string,
                epoch_start
            )
        )
    else:
        raise ValueError("Cannot train a negative amount.")

    return training_string


def build_data_string(data_set, reconstruction_distribution_name):

    if not data_set.noisy_preprocessing_methods:

        if data_set.preprocessing_methods:
            if data_set.preprocessing_methods == ["binarise"]:
                data_string = "binarised values"
            else:
                data_string = "preprocessed values"
        else:
            data_string = "original values"

        if reconstruction_distribution_name == "bernoulli":
            if not data_string == "binarised values":
                data_string += " with binarised values as targets"
        else:
            if not data_string == "original values":
                data_string += " with original values as targets"
    else:
        if data_set.noisy_preprocessing_methods == ["binarise"]:
            data_string = "new Bernoulli-sampled values"
        else:
            data_string = "new preprocessed values"
        data_string += " at every epoch"

    return data_string


def load_number_of_epochs_trained(model, run_id=None, early_stopping=False,
                                  best_model=False):

    n_epoch = None
    data_set_kind = "training"
    loss = "log_likelihood"
    loss_prefix = "losses/"

    if "VAE" in model.type:
        loss = "lower_bound"

    loss = loss_prefix + loss

    log_directory = model.log_directory(
        run_id=run_id,
        early_stopping=early_stopping,
        best_model=best_model
    )

    scalar_sets = _summary_reader(log_directory, data_set_kind, loss)

    if scalar_sets and data_set_kind in scalar_sets:
        data_set_scalars = scalar_sets[data_set_kind]
    else:
        data_set_scalars = None

    if data_set_scalars and loss in data_set_scalars:
        scalars = data_set_scalars[loss]
    else:
        scalars = None

    if scalars:
        n_epoch = max([scalar.step for scalar in scalars])

    return n_epoch


def load_learning_curves(model, data_set_kinds="all", run_id=None,
                         early_stopping=False, best_model=False,
                         log_directory=None):

    learning_curve_sets = {}

    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]

    if not log_directory:
        log_directory = model.log_directory(
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )

    if model.type == "GMVAE":
        losses = [
            "lower_bound",
            "reconstruction_error",
            "kl_divergence_z",
            "kl_divergence_y"
        ]
    elif "VAE" == model.type:
        losses = ["lower_bound", "reconstruction_error", "kl_divergence"]
    else:
        losses = ["log_likelihood"]

    loss_prefix = "losses/"
    loss_searches = list(map(lambda s: loss_prefix + s, losses))

    scalar_sets = _summary_reader(log_directory, data_set_kinds, loss_searches)

    for data_set_kind in data_set_kinds:

        learning_curve_set = {}

        for loss in losses:

            loss_tag = loss_prefix + loss

            if scalar_sets and data_set_kind in scalar_sets:
                data_set_scalars = scalar_sets[data_set_kind]
            else:
                data_set_scalars = None

            if data_set_scalars and loss_tag in data_set_scalars:
                scalars = data_set_scalars[loss_tag]
            else:
                scalars = None

            if scalars:

                learning_curve = numpy.empty(len(scalars))

                if len(scalars) == 1:
                    learning_curve[0] = scalars[0].value
                else:
                    for scalar in scalars:
                        learning_curve[scalar.step - 1] = scalar.value

            else:
                learning_curve = None

            learning_curve_set[loss] = learning_curve

        learning_curve_sets[data_set_kind] = learning_curve_set

    if len(data_set_kinds) == 1:
        learning_curve_sets = learning_curve_sets[data_set_kinds[0]]

    return learning_curve_sets


def load_accuracies(model, data_set_kinds="all", superset=False,
                    run_id=None, early_stopping=False, best_model=False):

    accuracies = {}

    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]

    log_directory = model.log_directory(
        run_id=run_id,
        early_stopping=early_stopping,
        best_model=best_model
    )

    accuracy_tag = "accuracy"

    if superset:
        accuracy_tag = "superset_" + accuracy_tag

    scalar_sets = _summary_reader(
        log_directory=log_directory,
        data_set_kinds=data_set_kinds,
        tag_searches=[accuracy_tag]
    )

    empty_scalar_sets = 0

    for data_set_kind in data_set_kinds:

        if scalar_sets and data_set_kind in scalar_sets:
            data_set_scalars = scalar_sets[data_set_kind]
        else:
            data_set_scalars = None

        if data_set_scalars and accuracy_tag in data_set_scalars:
            scalars = data_set_scalars[accuracy_tag]
        else:
            scalars = None

        if scalars:

            data_set_accuracies = numpy.empty(len(scalars))

            if len(scalars) == 1:
                data_set_accuracies[0] = scalars[0].value
            else:
                for scalar in scalars:
                    data_set_accuracies[scalar.step - 1] = scalar.value

            accuracies[data_set_kind] = data_set_accuracies

        else:
            accuracies[data_set_kind] = None
            empty_scalar_sets += 1

    if len(data_set_kinds) == 1:
        accuracies = accuracies[data_set_kinds[0]]

    if empty_scalar_sets == len(data_set_kinds):
        accuracies = None

    return accuracies


def load_centroids(model, data_set_kinds="all", run_id=None,
                   early_stopping=False, best_model=False):

    if "VAE" not in model.type:
        return None

    centroids_sets = {}

    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]

    log_directory = model.log_directory(
        run_id=run_id,
        early_stopping=early_stopping,
        best_model=best_model
    )

    centroid_tag = "cluster"

    scalar_sets = _summary_reader(
        log_directory=log_directory,
        data_set_kinds=data_set_kinds,
        tag_searches=[centroid_tag]
    )

    for data_set_kind in data_set_kinds:

        centroids_set = {}

        for distribution in ["prior", "posterior"]:

            cluster_tag = distribution + "/cluster_0/probability"

            if scalar_sets and data_set_kind in scalar_sets:
                data_set_scalars = scalar_sets[data_set_kind]
            else:
                data_set_scalars = None

            if data_set_scalars and cluster_tag in data_set_scalars:
                scalars = data_set_scalars[cluster_tag]
            else:
                scalars = None

            if not scalars:
                centroids_set[distribution] = None
                continue

            n_epoch = len(scalars)
            n_centroids = 1
            latent_size = model.latent_size

            if model.type == "GMVAE":
                n_centroids = model.number_of_latent_clusters

            z_probabilities = numpy.empty(shape=(n_epoch, n_centroids))
            z_means = numpy.empty(shape=(n_epoch, n_centroids, latent_size))
            z_variances = numpy.empty(
                shape=(n_epoch, n_centroids, latent_size))
            z_covariance_matrices = numpy.empty(
                shape=(n_epoch, n_centroids, latent_size, latent_size))

            for k in range(n_centroids):

                probability_scalars = data_set_scalars[
                    distribution + "/cluster_{}/probability".format(k)]

                if len(probability_scalars) == 1:
                    z_probabilities[0][k] = probability_scalars[0].value
                else:
                    for scalar in probability_scalars:
                        z_probabilities[scalar.step - 1][k] = scalar.value

                for l in range(latent_size):

                    mean_scalars = data_set_scalars[
                        distribution
                        + "/cluster_{}/mean/dimension_{}".format(k, l)
                    ]

                    if len(mean_scalars) == 1:
                        z_means[0][k][l] = mean_scalars[0].value
                    else:
                        for scalar in mean_scalars:
                            z_means[scalar.step - 1][k][l] = scalar.value

                    variance_scalars = data_set_scalars[
                        distribution
                        + "/cluster_{}/variance/dimension_{}".format(k, l)
                    ]

                    if len(variance_scalars) == 1:
                        z_variances[0][k][l] = variance_scalars[0].value
                    else:
                        for scalar in variance_scalars:
                            z_variances[scalar.step - 1][k][l] = scalar.value

                    if "full-covariance" in model.latent_distribution_name:
                        for l_ in range(latent_size):
                            covariance_scalars = data_set_scalars[
                                distribution
                                + "/cluster_{}/covariance/dimension_{}_{}"
                                .format(k, l, l_)
                            ]
                            if len(covariance_scalars) == 1:
                                z_covariance_matrices[0, k, l, l_] = (
                                    covariance_scalars[0].value)
                            else:
                                for scalar in covariance_scalars:
                                    z_covariance_matrices[
                                        scalar.step - 1, k, l, l_] = (
                                            scalar.value)

                if "full-covariance" not in model.latent_distribution_name:
                    for e in range(n_epoch):
                        z_covariance_matrices[e, k] = numpy.diag(
                            z_variances[e, k])

            if data_set_kind == "evaluation":
                z_probabilities = z_probabilities[0]
                z_means = z_means[0]
                z_covariance_matrices = z_covariance_matrices[0]

            centroids_set[distribution] = {
                "probabilities": z_probabilities,
                "means": z_means,
                "covariance_matrices": z_covariance_matrices
            }

        centroids_sets[data_set_kind] = centroids_set

    if len(data_set_kinds) == 1:
        centroids_sets = centroids_sets[data_set_kinds[0]]

    return centroids_sets


def load_kl_divergences(model, data_set_kind=None, run_id=None,
                        early_stopping=False, best_model=False):

    if data_set_kind is None:
        data_set_kind = "training"

    kl_neurons = None
    kl_divergence_neurons_tag_prefix = "kl_divergence_neurons/"
    log_directory = model.log_directory(
        run_id=run_id,
        early_stopping=early_stopping,
        best_model=best_model
    )

    scalar_sets = _summary_reader(
        log_directory=log_directory,
        data_set_kinds=data_set_kind,
        tag_searches=[kl_divergence_neurons_tag_prefix]
    )

    if scalar_sets and data_set_kind in scalar_sets:
        data_set_scalars = scalar_sets[data_set_kind]
    else:
        data_set_scalars = None

    kl_divergence_neuron_0_tag = kl_divergence_neurons_tag_prefix + "0"

    if data_set_scalars and kl_divergence_neuron_0_tag in data_set_scalars:
        scalars = data_set_scalars[kl_divergence_neuron_0_tag]
    else:
        scalars = None

    if scalars:

        n_epochs = len(scalars)

        if ("mixture" in model.latent_distribution_name
                and data_set_kind == "training"):
            latent_size = 1
        else:
            latent_size = model.latent_size

        kl_neurons = numpy.empty([n_epochs, latent_size])

        for i in range(latent_size):
            kl_divergence_neuron_i_tag = (
                kl_divergence_neurons_tag_prefix + str(i))

            if kl_divergence_neuron_i_tag in data_set_scalars:
                scalars = data_set_scalars[kl_divergence_neuron_i_tag]
            else:
                scalars = None

            if scalars:
                for scalar in scalars:
                    if data_set_kind == "training":
                        kl_neurons[scalar.step - 1, i] = scalar.value
                    else:
                        kl_neurons[0, i] = scalar.value
            else:
                kl_neurons[:, i] = numpy.full(n_epochs, numpy.nan)

    if kl_neurons is not None and kl_neurons.shape[0] == 1:
        kl_neurons = kl_neurons.squeeze(axis=0)

    return kl_neurons


def early_stopping_status(losses, early_stopping_rounds):

    n_epochs_without_improvements = 0
    stopped_early = False

    if losses is not None:

        n_epochs = len(losses)

        for epoch_number in range(1, n_epochs):

            if losses[epoch_number] < losses[epoch_number - 1]:
                n_epochs_without_improvements += 1
            else:
                n_epochs_without_improvements = 0

            if n_epochs_without_improvements >= early_stopping_rounds:
                stopped_early = True
                n_epochs_without_improvements = numpy.nan
                break

    return stopped_early, n_epochs_without_improvements


def better_model_exists(model, run_id=None):
    n_epochs_current = load_number_of_epochs_trained(
        model, run_id=run_id, best_model=False)
    n_epochs_best = load_number_of_epochs_trained(
        model, run_id=run_id, best_model=True)
    if n_epochs_best:
        better_model_exists = n_epochs_best < n_epochs_current
    else:
        better_model_exists = False
    return better_model_exists


def model_stopped_early(model, run_id=None):
    stopped_early, _ = model.early_stopping_status(run_id=run_id)
    return stopped_early


def generate_unique_run_id_for_model(model, timestamp=None):
    log_directory = model.log_directory()
    existing_run_ids = [
        re.sub(r"^run_", "", d)
        for d in os.listdir(log_directory)
        if d.startswith("run_")
    ]
    unique_run_id_found = False
    while not unique_run_id_found:
        run_id = _generate_run_id(timestamp=timestamp)
        if run_id not in existing_run_ids:
            unique_run_id_found = True
    return run_id


def check_run_id(run_id):
    if run_id is not None:
        run_id = str(run_id)
        if not re.fullmatch(r"[\w]+", run_id):
            raise ValueError(
                "`run_id` can only contain letters, numbers, and "
                "underscores ('_')."
            )
    else:
        raise TypeError("The run ID has not been set.")
    return run_id


def clear_log_directory(log_directory):
    remove_log_directory = True
    if os.path.exists(log_directory):
        for item_name in os.listdir(log_directory):
            item_path = os.path.join(log_directory, item_name)
            if os.path.isdir(item_path):
                if item_name.startswith("run_"):
                    remove_log_directory = False
                else:
                    shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        if remove_log_directory:
            shutil.rmtree(log_directory)


def correct_model_checkpoint_path(model_checkpoint_path, parent_directory):
    correct_model_checkpoint_path = os.path.join(
        parent_directory,
        os.path.basename(model_checkpoint_path)
    )
    return correct_model_checkpoint_path


def copy_model_directory(model_checkpoint, main_destination_directory):

    checkpoint_path_prefix = model_checkpoint.model_checkpoint_path
    checkpoint_directory, checkpoint_filename_prefix = (
        os.path.split(checkpoint_path_prefix))

    if not os.path.exists(main_destination_directory):
        os.makedirs(main_destination_directory)

    # Checkpoint file
    source_checkpoint_file_path = os.path.join(
        checkpoint_directory, "checkpoint")

    with open(source_checkpoint_file_path, "r") as checkpoint_file:
        source_checkpoint_line = checkpoint_file.readline()
    _, source_path_prefix = source_checkpoint_line.split(": ")
    source_path_prefix = source_path_prefix.replace("\"", "")

    if source_path_prefix.startswith("model"):
        destionation_checkpoint_path_prefix = checkpoint_filename_prefix
    else:
        destionation_checkpoint_path_prefix = os.path.join(
            main_destination_directory, checkpoint_filename_prefix)

    destionation_checkpoint_file_path = os.path.join(
        main_destination_directory, "checkpoint")

    with open(destionation_checkpoint_file_path, "w") as checkpoint_file:
        checkpoint_file.write(
            "model_checkpoint_path: \"{}\"".format(
                destionation_checkpoint_path_prefix) + "\n" +
            "all_model_checkpoint_paths: \"{}\"".format(
                destionation_checkpoint_path_prefix)
        )

    # Remaining
    for f in os.listdir(checkpoint_directory):
        source_path = os.path.join(checkpoint_directory, f)

        if checkpoint_filename_prefix in f or "events" in f:
            destination_directory = main_destination_directory
            shutil.copy(source_path, destination_directory)

        elif os.path.isdir(source_path):
            if f not in ["training", "validation"]:
                continue
            sub_checkpoint_directory = source_path
            destination_directory = os.path.join(main_destination_directory, f)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            for sub_f in os.listdir(sub_checkpoint_directory):
                sub_source_path = os.path.join(sub_checkpoint_directory, sub_f)
                shutil.copy(sub_source_path, destination_directory)


def remove_old_checkpoints(directory):

    checkpoint = tf.train.get_checkpoint_state(directory)

    if checkpoint:
        for f in os.listdir(directory):
            file_path = os.path.join(directory, f)
            is_old_checkpoint_file = (
                os.path.isfile(file_path)
                and "model" in f
                and checkpoint.model_checkpoint_path not in file_path
            )
            if is_old_checkpoint_file:
                os.remove(file_path)


def parse_model_versions(proposed_versions):

    version_alias_sets = {
        "end_of_training": ["eot", "end", "finish", "finished"],
        "best_model": ["bm", "best", "optimal_parameters", "op", "optimal"],
        "early_stopping": ["es", "early", "stop", "stopped"]
    }

    parsed_versions = []

    if not isinstance(proposed_versions, list):
        proposed_versions = [proposed_versions]

    if proposed_versions == ["all"]:
        parsed_versions = list(version_alias_sets.keys())

    else:
        for proposed_version in proposed_versions:

            normalised_proposed_version = normalise_string(proposed_version)
            parsed_version = None

            for version, version_aliases in version_alias_sets.items():
                if (normalised_proposed_version == version
                        or normalised_proposed_version in version_aliases):
                    parsed_version = version
                    break

            if parsed_version:
                parsed_versions.append(parsed_version)
            else:
                raise ValueError(
                    "`{}` is not a model version.".format(
                        proposed_version
                    )
                )

    return parsed_versions


def parse_numbers_of_samples(proposed_numbers_of_samples):
    required_scenarios = ["training", "evaluation"]

    if isinstance(proposed_numbers_of_samples, (int, float)):
        proposed_numbers_of_samples = [_parse_number_of_samples(
            proposed_numbers_of_samples)]

    if isinstance(proposed_numbers_of_samples, list):
        if len(proposed_numbers_of_samples) == 1:
            proposed_numbers_of_samples *= 2
        elif len(proposed_numbers_of_samples) > 2:
            raise ValueError(
                "List of number of samples can only contain one or two "
                "numbers."
            )
        parsed_numbers_of_samples = {
            scenario: _parse_number_of_samples(number)
            for scenario, number in zip(
                required_scenarios, proposed_numbers_of_samples)
        }

    elif isinstance(proposed_numbers_of_samples, dict):
        valid = True
        for scenario in required_scenarios:
            scenario_number = proposed_numbers_of_samples.get(scenario)
            try:
                scenario_number = _parse_number_of_samples(scenario_number)
            except TypeError as sample_number_parsing_error:
                if "integer" in sample_number_parsing_error:
                    valid = False
            finally:
                proposed_numbers_of_samples[scenario] = scenario_number
        if valid:
            parsed_numbers_of_samples = proposed_numbers_of_samples
        else:
            raise ValueError(
                "To supply the numbers of samples as a dictionary, the "
                "dictionary must contain the keys {} with the number of "
                "samples for each given as an integer.".format(
                    enumerate_strings(
                        ["`{}`".format(s) for s in required_scenarios],
                        conjunction="and"
                    )
                )
            )

    else:
        raise TypeError(
            "Expected an `int`, `list`, or `dict`; got `{}`.".format(
                type(proposed_numbers_of_samples))
        )

    return parsed_numbers_of_samples


def validate_model_parameters(reconstruction_distribution=None,
                              number_of_reconstruction_classes=None,
                              model_type=None, latent_distribution=None,
                              parameterise_latent_posterior=None):

    # Validate piecewise categorical likelihood
    if reconstruction_distribution and number_of_reconstruction_classes:
        if number_of_reconstruction_classes > 0:
            piecewise_categorical_likelihood_errors = []

            if reconstruction_distribution == "bernoulli":
                piecewise_categorical_likelihood_errors.append(
                    "the Bernoulli distribution")

            if "zero-inflated" in reconstruction_distribution:
                piecewise_categorical_likelihood_errors.append(
                    "zero-inflated distributions")

            if "constrained" in reconstruction_distribution:
                piecewise_categorical_likelihood_errors.append(
                    "constrained distributions")

            if len(piecewise_categorical_likelihood_errors) > 0:
                piecewise_categorical_likelihood_error = (
                    "{} cannot be piecewise categorical.".format(
                        capitalise_string(
                            enumerate_strings(
                                piecewise_categorical_likelihood_errors,
                                conjunction="or"
                            )
                        )
                    )
                )
                raise ValueError(piecewise_categorical_likelihood_error)

    # Validate parameterisation of latent posterior for VAE
    if model_type and latent_distribution and parameterise_latent_posterior:
        if "VAE" in model_type:
            if (not (model_type in ["VAE"]
                     and latent_distribution == "gaussian mixture")
                    and parameterise_latent_posterior):

                parameterise_error = (
                    "Cannot parameterise latent posterior parameters for {} "
                    "or {} distribution.".format(
                        model_type, latent_distribution)
                )
                raise ValueError(parameterise_error)


def batch_indices_for_subset(subset):
    batch_indices = subset.batch_indices
    if batch_indices is None:
        raise TypeError(
            "No batch indices found in {} set.".format(subset.kind))
    return batch_indices


def _summary_reader(log_directory, data_set_kinds, tag_searches):

    scalars = None
    ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])

    if not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]

    if not isinstance(tag_searches, list):
        tag_searches = [tag_searches]

    if os.path.exists(log_directory):

        scalars = {}

        for data_set_kind in data_set_kinds:
            data_set_log_directory = os.path.join(log_directory, data_set_kind)
            data_set_scalars = None
            if os.path.exists(data_set_log_directory):
                data_set_scalars = {}
                for filename in sorted(os.listdir(data_set_log_directory)):
                    if filename.startswith("event"):
                        events_path = os.path.join(
                            data_set_log_directory,
                            filename
                        )
                        events = tf.train.summary_iterator(
                            events_path)
                        for event in events:
                            for value in event.summary.value:
                                for tag_search in tag_searches:
                                    if tag_search in value.tag:
                                        tag = value.tag
                                        scalar = ScalarEvent(
                                            wall_time=event.wall_time,
                                            step=event.step,
                                            value=value.simple_value
                                        )
                                        if tag not in data_set_scalars:
                                            data_set_scalars[tag] = []
                                        data_set_scalars[tag].append(scalar)
            scalars[data_set_kind] = data_set_scalars

    return scalars


def _generate_run_id(timestamp=None, number_of_letters=2):

    if timestamp is None:
        timestamp = time.time()

    formatted_timestamp = datetime.utcfromtimestamp(timestamp).strftime(
        "%Y%m%dT%H%M%SZ")

    uppercase_ascii_letters = list(ascii_uppercase)
    letters = "".join(random.choices(
        population=uppercase_ascii_letters,
        k=number_of_letters
    ))

    run_id = formatted_timestamp + "_" + letters

    return run_id


def _parse_number_of_samples(number):
    if isinstance(number, int):
        pass
    elif (isinstance(number, float)
            and number.is_integer()):
        number = int(number)
    else:
        raise TypeError("Number of samples should be an integer.")
    return number
