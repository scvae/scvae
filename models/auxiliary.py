import tensorflow as tf

from tensorflow.contrib.layers import (
    fully_connected, batch_norm, dropout,
    variance_scaling_initializer, xavier_initializer
)

from tensorflow.python.ops.nn import relu

import os, shutil

## N(mu=0,sigma=sqrt(2/n_in)) weight and 0-bias initialiser.
# weights_init = variance_scaling_initializer(factor=2.0, mode ='FAN_IN', 
#     uniform = False, seed = None, dtype = tf.float32)
weights_init = xavier_initializer()

def prelu(inputs, is_training, scope):
    with tf.variable_scope(scope):
        a = tf.Variable(0.25*tf.ones([inputs.shape[-1]]), name="a")
        return tf.max(0, inputs) + tf.multiply(a, tf.min(0, inputs))


# Wrapper layer for inserting batch normalization in between linear and nonlinear activation layers.
def dense_layer(inputs, num_outputs, is_training = True, scope = "layer", 
    activation_fn = None, batch_normalisation = False, decay = 0.999, 
    center = True, scale = False, reuse = False, 
    dropout_keep_probability = False):
    
    with tf.variable_scope(scope): 
        # Dropout input connections with rate = (1- dropout_keep_probability)
        if dropout_keep_probability and dropout_keep_probability != 1:
            inputs = dropout(inputs, 
                keep_prob = dropout_keep_probability, 
                is_training = is_training
            )

        # Set up weights for and transform inputs through neural network. 
        outputs = fully_connected(inputs,
            num_outputs = num_outputs,
            activation_fn = None,
            weights_initializer = weights_init, 
            scope = 'DENSE',
            reuse = reuse
        )

        # Set up normalisation across examples with learned center and scale. 
        if batch_normalisation:
            outputs = batch_norm(outputs,
                center = center,
                scale = scale,
                is_training = is_training,
                scope = 'BATCH_NORM',
                reuse = reuse
            )

        # Apply non-linear activation function to linear outputs
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    
    return outputs

# Wrapper layer for inserting batch normalization in between several linear
# and non-linear activation layers in given or reverse order of num_outputs.
def dense_layers(inputs, num_outputs, reverse_order = False, is_training = True,
    scope = "layers", activation_fn = None, batch_normalisation = False, 
    decay = 0.999, center = True, scale = False, reuse = False, 
    input_dropout_keep_probability = False,
    hidden_dropout_keep_probability = False):
    if not isinstance(num_outputs, (list, tuple)):
        num_outputs = [num_outputs]
    if reverse_order:
        num_outputs = num_outputs[::-1]
    outputs = inputs
    # Set up all following layers
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
                inputs = outputs,
                num_outputs = num_output,
                is_training = is_training,
                scope = 'LAYER_{:d}'.format(layer_number),
                activation_fn = activation_fn,
                batch_normalisation = batch_normalisation,
                decay = decay,
                center = center,
                scale = scale,
                reuse = reuse,
                dropout_keep_probability = dropout_keep_probability
            )
    
    return outputs

def log_reduce_exp(A, reduction_function=tf.reduce_mean, axis=None):
    # log-mean-exp over axis to avoid overflow and underflow
    A_max = tf.reduce_max(A, axis=axis, keep_dims=True)
    B = tf.log(reduction_function(
        tf.exp(A - A_max), axis = axis, keep_dims=True)) + A_max
    return tf.squeeze(B)

def reduce_logmeanexp(input_tensor, axis=None, keep_dims=False):
    """Computes log(mean(exp(elements across dimensions of a tensor))).

    Parameters
    ----------
    input_tensor : tf.Tensor
    The tensor to reduce. Should have numeric type.
    axis : int or list of int, optional
    The dimensions to reduce. If `None` (the default), reduces all
    dimensions.
    keep_dims : bool, optional
    If true, retains reduced dimensions with length 1.

    Returns
    -------
    tf.Tensor
    The reduced tensor.
    """
    logsumexp = tf.reduce_logsumexp(input_tensor, axis, keep_dims)
    input_tensor = tf.convert_to_tensor(input_tensor)
    n = input_tensor.get_shape().as_list()
    if axis is None:
        n = tf.cast(tf.reduce_prod(n), logsumexp.dtype)
    else:
        n = tf.cast(tf.reduce_prod(n[axis]), logsumexp.dtype)

    return -tf.log(n) + logsumexp

def pairwise_distance(a, b = None):
    if not b:
        r = tf.reduce_sum(a*a, axis = 1, keep_dims=True)
        D = r - 2*tf.matmul(a, a, transpose_b = True)\
            + tf.transpose(r)
    else:
        r_a = tf.reduce_sum(a*a, 1, keep_dims=True)
        r_b = tf.reshape(tf.reduce_sum(b*b, axis = 1, keep_dims=True), [1, -1])
        D = r_a - 2*tf.matmul(a, b, transpose_b=True) + r_b
    return D

# Early stopping

def epochsWithNoImprovement(losses):
    
    E = len(losses)
    k = 0
    
    for e in range(1, E):
        if losses[e] < losses[e - 1]:
            k += 1
        else:
            k = 0
    
    return k    

# Strings

def trainingString(epoch_start, number_of_epochs, data_string):
    
    if epoch_start == 0:
        training_string = "Training model for {} epochs on {}.".format(
            number_of_epochs, data_string)
    elif epoch_start < number_of_epochs:
        training_string = "Continue training model for {}".format(
            number_of_epochs - epoch_start) + " additionally epochs" \
            + " (up to {} epochs)".format(number_of_epochs) \
            + " on {}.".format(data_string)
    elif epoch_start == number_of_epochs:
        training_string = "Model has already been trained for" \
            + " {} epochs on {}.".format(number_of_epochs, data_string)
    elif epoch_start > number_of_epochs:
        training_string = "Model has already been trained for more" \
            + " than {} epochs on {}.".format(number_of_epochs,
                data_string) \
            + " Loading model trained for {} epochs.".format(
                epoch_start)
    else:
        raise ValueError("Cannot train a negative amount.")
    
    return training_string

def dataString(data_set, reconstruction_distribution_name):
    
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

# Model

def copyModelDirectory(model_checkpoint, main_destination_directory):
    
    checkpoint_path_prefix = model_checkpoint.model_checkpoint_path
    checkpoint_directory, checkpoint_filename_prefix = \
        os.path.split(checkpoint_path_prefix)
    
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
    
    # The rest
    
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

def removeOldCheckpoints(directory):
    
    checkpoint = tf.train.get_checkpoint_state(directory)
    
    print(checkpoint.model_checkpoint_path)
    
    if checkpoint:
        for f in os.listdir(directory):
            file_path = os.path.join(directory, f)
            is_old_checkpoint_file = os.path.isfile(file_path) \
                and "model" in f \
                and not checkpoint.model_checkpoint_path in file_path
            if is_old_checkpoint_file:
                os.remove(file_path)
