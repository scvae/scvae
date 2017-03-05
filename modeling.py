import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.python.ops.nn import relu
from tensorflow import sigmoid, identity
from tensorflow.contrib.distributions import Bernoulli, Normal, Poisson, Categorical, kl
from distributions import (
    ZeroInflatedPoisson, NegativeBinomial, ZeroInflatedNegativeBinomial, ZeroInflated, Categorized, Pareto
)

import os

import numpy

from numpy import inf

from time import time

eps = 1e-6

class VariationalAutoEncoder(object):
    def __init__(self, feature_size, latent_size, hidden_sizes,
        reconstruction_distribution = None, number_of_reconstruction_classes = None,
        use_batch_norm = True, use_count_sum = True, epsilon = 1e-6):
        
        # Setup
        
        super(VariationalAutoEncoder, self).__init__()
        
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = distributions[reconstruction_distribution]
        
        self.k_max = number_of_reconstruction_classes
        
        self.use_batch_norm = use_batch_norm
        self.use_count_sum = use_count_sum
        
        self.epsilon = epsilon
        
        # self.graph = tf.Graph()
        
        self.x = tf.placeholder(tf.float32, [None, self.feature_size], 'x') # counts

        if self.use_count_sum:
            self.n = tf.placeholder(tf.float32, [None, 1], 'N') # total counts sum

        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        self.warm_up_weight = tf.placeholder(tf.float32, [], 'warm_up_weight')

        self.is_training = tf.placeholder(tf.bool, [], 'phase')

        self.inference()
        self.loss()
        self.training()

        self.summary = tf.summary.merge_all()

        for parameter in tf.trainable_variables():
            print(parameter.name, parameter.get_shape())
    
    @property
    def name(self):

        #model_name = dataSetBaseName(splitting_method, splitting_fraction,
            #filtering_method, feature_selection, feature_size)
        
        model_name = self.reconstruction_distribution_name.replace(" ", "_")
        
        # if self.k_max:
        #     model_name += "_c_" + str(self.k_max)
        
        if self.use_count_sum:
            model_name += "_sum"
        
        model_name += "_l_" + str(self.latent_size) + "_h_" + "_".join(map(str,self.hidden_sizes))
        
        if self.use_batch_norm:
            model_name += "_bn"
        
        # model_name += "_lr_{:.1g}".format(self.learning_rate)
        # model_name += "_b_" + str(self.batch_size)
        # model_name += "_wu_" + str(number_of_warm_up_epochs)
        
        # model_name += "_e_" + str(number_of_epochs)
        
        return model_name

    def inference(self):
        
        encoder = self.x
        
        with tf.variable_scope("ENCODER"):
            for i, hidden_size in enumerate(self.hidden_sizes):
                encoder = dense_layer(
                    inputs = encoder,
                    num_outputs = hidden_size,
                    activation_fn = relu,
                    use_batch_norm = self.use_batch_norm, 
                    is_training = self.is_training,
                    scope = '{:d}'.format(i + 1)
                )
        
        with tf.variable_scope("Z"):
            z_mu = dense_layer(
                inputs = encoder,
                num_outputs = self.latent_size,
                activation_fn = None,
                use_batch_norm = False,
                is_training = self.is_training,
                scope = 'MU')
            
            z_sigma = dense_layer(
                inputs = encoder,
                num_outputs = self.latent_size,
                activation_fn = lambda x: tf.exp(tf.clip_by_value(x, -3, 3)),
                use_batch_norm = False,
                is_training = self.is_training,
                scope = 'SIGMA')
            
            self.q_z_given_x = Normal(mu = z_mu, sigma = z_sigma)
            
            # Mean of z
            self.z_mean = self.q_z_given_x.mean()
        
            # Stochastic layer
            self.z = self.q_z_given_x.sample()
        
        # Decoder - Generative model, p(x|z)
        
        if self.use_count_sum:
            decoder = tf.concat([self.z, self.n], axis = 1, name = 'Z_N')
        else:
            decoder = self.z
        
        with tf.variable_scope("DECODER"):
            for i, hidden_size in enumerate(reversed(self.hidden_sizes)):
                decoder = dense_layer(
                    inputs = decoder,
                    num_outputs = hidden_size,
                    activation_fn = relu,
                    use_batch_norm = self.use_batch_norm,
                    is_training = self.is_training,
                    scope = '{:d}'.format(len(self.hidden_sizes) - i)
                )

        # Reconstruction distribution parameterisation
        
        with tf.variable_scope("X_TILDE"):
            
            x_theta = {}
        
            for parameter in self.reconstruction_distribution["parameters"]:
                
                parameter_activation_function = \
                    self.reconstruction_distribution["parameters"]\
                    [parameter]["activation function"]
                p_min, p_max = \
                    self.reconstruction_distribution["parameters"]\
                    [parameter]["support"]
                
                x_theta[parameter] = dense_layer(
                    inputs = decoder,
                    num_outputs = self.feature_size,
                    activation_fn = lambda x: tf.clip_by_value(
                        parameter_activation_function(x),
                        p_min + self.epsilon,
                        p_max - self.epsilon
                    ),
                    is_training = self.is_training,
                    scope = parameter.upper()
                )
        
            self.p_x_given_z = self.reconstruction_distribution["class"](x_theta)
        
            if self.k_max:
                
                x_logits = dense_layer(
                    inputs = decoder,
                    num_outputs = self.feature_size * self.k_max,
                    activation_fn = None,
                    is_training = self.is_training,
                    scope = "P_K"
                )
                
                x_logits = tf.reshape(x_logits, [-1, self.feature_size, self.k_max])
                
                self.p_x_given_z = Categorized(
                    dist = self.p_x_given_z,
                    cat = Categorical(logits = x_logits)
                )
            
            self.x_tilde_mean = self.p_x_given_z.mean()
        
        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            tf.summary.histogram(parameter.name, parameter)
    
    def loss(self):
        
        # Recognition prior
        p_z_mu = tf.constant(0.0, dtype = tf.float32)
        p_z_sigma = tf.constant(1.0, dtype = tf.float32)
        p_z = Normal(p_z_mu, p_z_sigma)
        
        # Loss
        
        ## Reconstruction error
        log_p_x_given_z = tf.reduce_mean(
            tf.reduce_sum(self.p_x_given_z.log_prob(self.x), axis = 1),
            name = 'reconstruction_error'
        )
        tf.add_to_collection('losses', log_p_x_given_z)
        
        ## Regularisation
        KL_qp = tf.reduce_mean(
            tf.reduce_sum(kl(self.q_z_given_x, p_z), axis = 1),
            name = "kl_divergence"
        )
        tf.add_to_collection('losses', KL_qp)

        # Averaging over samples.
        self.loss_op = tf.subtract(log_p_x_given_z, KL_qp, name = 'lower_bound')
        tf.add_to_collection('losses', self.loss_op)
        
        # Add scalar summaries for the losses
        for l in tf.get_collection('losses'):
            tf.summary.scalar(l.op.name, l)
    
    def training(self):
        
        # Create the gradient descent optimiser with the given learning rate.
        def setupTraining():
            
            # Optimizer and training objective of negative loss
            optimiser = tf.train.AdamOptimizer(self.learning_rate)
            
            # Create a variable to track the global step.
            self.global_step = tf.Variable(0, name = 'global_step',
                trainable = False)
            
            # Use the optimiser to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training
            # step.
            self.train_op = optimiser.minimize(
                -self.loss_op,
                global_step = self.global_step
            )
        
        # Make sure that the updates of the moving_averages in batch_norm
        # layers are performed before the train_step.
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                setupTraining()
        else:
            setupTraining()

    def train(self, train_data, valid_data, number_of_epochs = 50, batch_size = 100, learning_rate = 1e-3, log_directory = None, reset_training = False):

        if self.use_count_sum:
            n_train = train_data.counts.sum(axis = 1).reshape(-1, 1)
            n_valid = valid_data.counts.sum(axis = 1).reshape(-1, 1)

        if reset_training and os.path.exists(log_directory):
            for f in os.listdir(log_directory):
                os.remove(os.path.join(log_directory, f))
            os.rmdir(log_directory)

        # Train
        
        M = train_data.number_of_examples

        self.saver = tf.train.Saver()
        checkpoint_file = os.path.join(log_directory, 'model.ckpt')
        
        with tf.Session() as session:
        
            summary_writer = tf.summary.FileWriter(log_directory, session.graph)
        
            session.run(tf.global_variables_initializer())

            # Print out the defined graph
            # print("The inference graph:")
            # print(tf.get_default_graph().as_graph_def())

            #train_losses, valid_losses = [], []
            feed_dict_train = {self.x: train_data.counts, self.is_training: False}
            feed_dict_valid = {self.x: valid_data.counts, self.is_training: False}
            if self.use_count_sum:
                feed_dict_train[self.n] = n_train
                feed_dict_valid[self.n] = n_valid
            
            for epoch in range(number_of_epochs):
                shuffled_indices = numpy.random.permutation(M)
                for i in range(0, M, batch_size):

                    step = session.run(self.global_step)

                    start_time = time()

                    # Feeding in batch to model
                    subset = shuffled_indices[i:(i + batch_size)]
                    batch = train_data.counts[subset]
                    feed_dict_batch = {self.x: batch, self.is_training: True, self.learning_rate: learning_rate}

                    # Adding the sum of counts per cell to the generator after the sample layer.
                    if self.use_count_sum:
                        feed_dict_batch[self.n] = n_train[subset]

                    # Run the stochastic batch training operation.
                    _, batch_loss = session.run([self.train_op, self.loss_op], feed_dict = feed_dict_batch)

                    # Duration of one training step.
                    duration = time() - start_time

                    # Evaluation printout and TensorBoard summary
                    if step % 10 == 0:
                        print('Step {:d}: loss = {:.2f} ({:.3f} sec)'.format(int(step), batch_loss, duration))
                        summary_str = session.run(self.summary, feed_dict = feed_dict_batch)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                # Saving model parameters
                print('Checkpoint reached: Saving model')
                self.saver.save(session, checkpoint_file)
                print('Done saving model')

                # Evaluation
                print('Evaluating epoch {:d}'.format(epoch))
            
                train_loss = 0
                for i in range(0, M, batch_size):
                    subset = slice(i, (i + batch_size))
                    batch = train_data.counts[subset]
                    feed_dict_batch = {self.x: batch, self.is_training: False}
                    if self.use_count_sum:
                        feed_dict_batch[self.n] = n_train[subset]
                    train_loss += session.run(self.loss_op, feed_dict = feed_dict_batch)
                train_loss /= M / batch_size
                print('Done evaluating training set')
            
                valid_loss = 0
                for i in range(0, valid_data.number_of_examples, batch_size):
                    subset = slice(i, (i + batch_size))
                    batch = valid_data.counts[subset]
                    feed_dict_batch = {self.x: batch, self.is_training: False}
                    if self.use_count_sum:
                        feed_dict_batch[self.n] = n_valid[subset]
                    valid_loss += session.run(self.loss_op, feed_dict = feed_dict_batch)
                valid_loss /= valid_data.number_of_examples / batch_size
                print('Done evaluating validation set')
            
                print("Epoch %d: ELBO: %g (Train), %g (Valid)"%(epoch+1, train_loss, valid_loss))


    def evaluate(self, test_set, batch_size = 100, log_directory = None):

        checkpoint = tf.train.get_checkpoint_state(log_directory)
        
        if self.use_count_sum:
            n_test = test_set.counts.sum(axis = 1).reshape(-1, 1)
                
        with tf.Session() as session:
        
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(session, checkpoint.model_checkpoint_path)
        
            lower_bound_test = 0
            recon_mean_test = numpy.empty([test_set.number_of_examples, test_set.number_of_features])
            z_mu_test = numpy.empty([test_set.number_of_examples, self.latent_size])
            for i in range(0, test_set.number_of_examples, batch_size):
                subset = slice(i, (i + batch_size))
                batch = test_set.counts[subset]
                feed_dict_batch = {self.x: batch, self.is_training: False}
                if self.use_count_sum:
                    feed_dict_batch[self.n] = n_test[subset]
                lower_bound_batch, recon_mean_batch, z_mu_batch = session.run([self.loss_op, self.x_tilde_mean, self.z_mean], feed_dict = feed_dict_batch)
                lower_bound_test += lower_bound_batch
                recon_mean_test[subset] = recon_mean_batch
                z_mu_test[subset] = z_mu_batch
            lower_bound_test /= test_set.number_of_examples / batch_size

            metrics_test = {
                "LL_test": lower_bound_test
            }
        
            print(metrics_test)
        
            return recon_mean_test, z_mu_test, metrics_test

distributions = {
    "bernoulli": {
        "parameters": {
            "p": {
                "support": [0, 1],
                "activation function": sigmoid
            }
        },
        "class": lambda theta: Bernoulli(
            p = theta["p"]
        )
    },
    
    "gauss": {
        "parameters": {
            "mu": {
                "support": [-inf, inf],
                "activation function": identity
            },
            "log_sigma": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: Normal(
            mu = theta["mu"],
            sigma = tf.exp(theta["log_sigma"])
        )
    },
    
    "poisson": {
        "parameters": {
            "log_lambda": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: Poisson(
            lam = tf.exp(theta["log_lambda"])
        )
    },

    "pareto": {
        "parameters": {
            "log_sigma": {
                "support": [-10, 10],
                "activation function": identity
            },
            "log_alpha": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: Pareto(
            sigma = tf.exp(theta["log_sigma"]),
            alpha = tf.exp(theta["log_alpha"])
        )
    },
    
    "zero-inflated poisson": {
        "parameters": {
            "pi": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "log_lambda": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: ZeroInflated(
            Poisson(
                lam = tf.exp(theta["log_lambda"])
            ),
            pi = theta["pi"]
        )
    },
    
    "negative binomial": {
        "parameters": {
            "p": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "log_r": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: NegativeBinomial(
            p = theta["p"],
            r = tf.exp(theta["log_r"])
        )
    },
    
    "zero-inflated negative binomial": {
        "parameters": {
            "pi": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "p": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "log_r": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: ZeroInflated(
            NegativeBinomial(
                p = theta["p"],
                r = tf.exp(theta["log_r"])
            ),
            pi = theta["pi"]
        )
    }
}


# Wrapper layer for inserting batch normalization in between linear and nonlinear activation layers.
def dense_layer(inputs, num_outputs, is_training, scope, activation_fn = None,
    use_batch_norm = False, decay = 0.999, center = True, scale = False):
    
    with tf.variable_scope(scope):
        outputs = fully_connected(inputs, num_outputs = num_outputs, activation_fn = None, scope = 'DENSE')
        if use_batch_norm:
            outputs = batch_norm(outputs, center = center, scale = scale, is_training = is_training, scope = 'BATCH_NORM')
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    
    return outputs
