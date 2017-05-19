import tensorflow as tf

from tensorflow.contrib.layers import (
    fully_connected, batch_norm,
    variance_scaling_initializer, xavier_initializer
)

from tensorflow.python.ops.nn import relu

## N(mu=0,sigma=sqrt(2/n_in)) weight and 0-bias initialiser.
weights_init = variance_scaling_initializer(factor=2.0, mode ='FAN_IN', 
    uniform = False, seed = None, dtype = tf.float32)

def prelu(inputs, is_training, scope):
    with tf.variable_scope(scope):
        a = tf.Variable(0.25*tf.ones([inputs.shape[-1]]), name="a")
        return tf.max(0, inputs) + tf.multiply(a, tf.min(0, inputs))


# Wrapper layer for inserting batch normalization in between linear and nonlinear activation layers.
def dense_layer(inputs, num_outputs, is_training, scope, activation_fn = None,
    batch_normalisation = False, decay = 0.999, center = True, scale = False):
    
    with tf.variable_scope(scope):
        if activation_fn == relu: 
            # For relu use:
            ## N(mu=0,sigma=sqrt(2/n_in)) weight initialization
            # weights_init = variance_scaling_initializer(factor=2.0,
            #     mode ='FAN_IN', uniform = False, seed = None, dtype = tf.float32)
            # and 0 bias initialization.
            weights_init = xavier_initializer()
            outputs = fully_connected(inputs, num_outputs = num_outputs, activation_fn = None, weights_initializer = weights_init, scope = 'DENSE')
        else:
            # For all other activation functions use (the same):
            ## N(mu=0,sigma=sqrt(2/n_in) weight initialization
            # weights_init = variance_scaling_initializer(factor=2.0,
            #     mode = 'FAN_IN', uniform = False, seed = None, dtype = tf.float32)
            ## and 0 bias initialization.
            weights_init = xavier_initializer()
            outputs = fully_connected(inputs, num_outputs = num_outputs, activation_fn = None, weights_initializer = weights_init, scope = 'DENSE')
        if batch_normalisation:
            outputs = batch_norm(outputs, center = center, scale = scale, is_training = is_training, scope = 'BATCH_NORM')
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    
    return outputs

# Wrapper layer for inserting batch normalization in between linear and nonlinear activation layers.
def dense_layers(inputs, num_outputs, is_training, scope, activation_fn = None,
    batch_normalisation = False, decay = 0.999, center = True, scale = False):
    if not isinstance(num_outputs, list):
        num_outputs = [num_outputs]
    # Set up all following layers
    for i, num_output in enumerate(num_outputs):
        with tf.variable_scope('{:d}'.format(i + 1)):
            outputs = fully_connected(inputs, num_outputs = num_output,
                activation_fn = None, weights_initializer = weights_init, 
                scope = 'DENSE')
            if batch_normalisation:
                outputs = batch_norm(outputs, center = center, scale = scale,   
                    is_training = is_training, scope = 'BATCH_NORM')
            if activation_fn is not None:
                outputs = activation_fn(outputs)
    
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
