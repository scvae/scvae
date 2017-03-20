import tensorflow as tf

from tensorflow.contrib.layers import (
    fully_connected, batch_norm, variance_scaling_initializer
)

from tensorflow.python.ops.nn import relu

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
            weights_init = variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
            # and 0 bias initialization.
            outputs = fully_connected(inputs, num_outputs = num_outputs, activation_fn = None, weights_initializer = weights_init, scope = 'DENSE')
        else:
            # For all other activation functions use (the same):
            ## N(mu=0,sigma=sqrt(2/n_in) weight initialization
            weights_init = variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
            ## and 0 bias initialization.
            outputs = fully_connected(inputs, num_outputs = num_outputs, activation_fn = None, weights_initializer = weights_init, scope = 'DENSE')
        if batch_normalisation:
            outputs = batch_norm(outputs, center = center, scale = scale, is_training = is_training, scope = 'BATCH_NORM')
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    
    return outputs
