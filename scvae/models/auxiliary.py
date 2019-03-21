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
from datetime import datetime
from string import ascii_uppercase

import numpy
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout

from auxiliary import capitaliseString


# Wrapper layer for inserting batch normalisation in between linear and
# nonlinear activation layers
def dense_layer(inputs, num_outputs, is_training=True, scope="layer",
                activation_fn=None, batch_normalisation=False, decay=0.999,
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
        if batch_normalisation:
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
                 batch_normalisation=False, decay=0.999, center=True,
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
                batch_normalisation=batch_normalisation,
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
                capitaliseString(model_string), number_of_epochs, data_string
            )
        )
    elif epoch_start > number_of_epochs:
        training_string = (
            "{} has already been trained for more than {} epochs on {}. "
            "Loading model trained for {} epochs.".format(
                capitaliseString(model_string),
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
