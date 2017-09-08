import tensorflow as tf

from models.auxiliary import (
    dense_layer, dense_layers,
    log_reduce_exp, reduce_logmeanexp,
    trainingString, dataString
)

from tensorflow.python.ops.nn import relu, softmax, softplus
from tensorflow import sigmoid, identity

from tensorflow.contrib.distributions import Normal, Bernoulli,kl_divergence, Categorical
from distributions import distributions, latent_distributions, Categorized

import numpy
from numpy import inf

import copy
import os, shutil
from time import time
from auxiliary import formatDuration, normaliseString

from data import DataSet
from analysis import analyseIntermediateResults


class ClusterVariationalAutoEncoder(object):
    def __init__(self, feature_size, latent_size, hidden_sizes,
        number_of_monte_carlo_samples, number_of_importance_samples,
        number_of_latent_clusters = 10,
        reconstruction_distribution = None,
        number_of_reconstruction_classes = None,
        batch_normalisation = True, 
        dropout_keep_probability = False,
        count_sum = True,
        number_of_warm_up_epochs = 0, epsilon = 1e-6,
        log_directory = "log", results_directory = "results"):
        
        # Class setup
        super(ClusterVariationalAutoEncoder, self).__init__()
        
        self.type = "CVAE"
        
        self.Dim_x = feature_size
        self.Dim_z = latent_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        self.latent_distribution_name = "gaussian mixture"
        self.latent_distribution = copy.deepcopy(
            latent_distributions[self.latent_distribution_name]
        )
        self.Dim_y = number_of_latent_clusters
        self.number_of_latent_clusters = number_of_latent_clusters
        
        # Dictionary holding number of samples needed for the "monte carlo" 
        # estimator and "importance weighting" during both "train" and "test" time.  
        self.number_of_importance_samples = number_of_importance_samples
        self.number_of_monte_carlo_samples = number_of_monte_carlo_samples

        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = distributions\
            [reconstruction_distribution]
        
        self.k_max = number_of_reconstruction_classes
        
        self.batch_normalisation = batch_normalisation
        self.dropout_keep_probability = dropout_keep_probability

        self.count_sum_feature = count_sum
        self.count_sum = self.count_sum_feature or "constrained" in \
            self.reconstruction_distribution_name or "multinomial" in \
            self.reconstruction_distribution_name

        self.number_of_warm_up_epochs = number_of_warm_up_epochs

        self.epsilon = epsilon
        self.log_sigma_support = lambda x: tf.clip_by_value(x, -3 + self.epsilon, 3 - self.epsilon)
        
        self.main_log_directory = log_directory
        self.main_results_directory = results_directory

        # Graph setup
        
        self.graph = tf.Graph()
        
        self.parameter_summary_list = []
        
        with self.graph.as_default():
            
            self.x = tf.placeholder(tf.float32, [None, self.Dim_x], 'X')
            self.t = tf.placeholder(tf.float32, [None, self.Dim_x], 'T')
            
            if self.count_sum:
                self.n = tf.placeholder(tf.float32, [None, 1], 'N')
            
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            
            self.warm_up_weight = tf.placeholder(tf.float32, [], 'warm_up_weight')
            parameter_summary = tf.summary.scalar('warm_up_weight',
                self.warm_up_weight)
            self.parameter_summary_list.append(parameter_summary)
            
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')
            
            self.S_iw = tf.placeholder(
                tf.int32,
                [],
                'number_of_iw_samples'
            )
            self.S_mc = tf.placeholder(
                tf.int32,
                [],
                'number_of_mc_samples'
            )

            self.inference()
            self.loss()
            self.training()
            
            self.saver = tf.train.Saver(max_to_keep = 1)
    
    @property
    def training_name(self):
        return self.name("training")
    
    @property
    def testing_name(self):
        return self.name("testing")
    
    def name(self, process):
        
        latent_part = normaliseString(self.latent_distribution_name)
        
        if "mixture" in self.latent_distribution_name:
            latent_part += "_c_" + str(self.Dim_y)
        
        reconstruction_part = normaliseString(
            self.reconstruction_distribution_name)
        
        if self.k_max:
            reconstruction_part += "_c_" + str(self.k_max)
        
        if self.count_sum_feature:
            reconstruction_part += "_sum"
        
        reconstruction_part += "_l_" + str(self.Dim_z) \
            + "_h_" + "_".join(map(str, self.hidden_sizes))
        
        mc_train = self.number_of_monte_carlo_samples["training"]
        mc_eval = self.number_of_monte_carlo_samples["evaluation"]
        
        reconstruction_part += "_mc_" + str(mc_train)
        if process == "testing":
            reconstruction_part += "_" + str(mc_eval)
        
        iw_train = self.number_of_importance_samples["training"]
        iw_eval = self.number_of_importance_samples["evaluation"]
        
        reconstruction_part += "_iw_" + str(iw_train)
        if process == "testing":
            reconstruction_part += "_" + str(iw_eval)
        
        if self.batch_normalisation:
            reconstruction_part += "_bn"

        if self.dropout_keep_probability and \
        self.dropout_keep_probability != 1:
            reconstruction_part += "_do_" + str(self.dropout_keep_probability)
        
        if self.number_of_warm_up_epochs:
            reconstruction_part += "_wu_" + str(self.number_of_warm_up_epochs)
        
        model_name = os.path.join(self.type, latent_part, reconstruction_part)
        
        return model_name
    
    @property
    def log_directory(self):
        return os.path.join(self.main_log_directory, self.training_name)
    
    @property
    def title(self):
        
        title = model.type
        
        configuration = [
            self.reconstruction_distribution_name.capitalize(),
            "$l = {}$".format(self.Dim_z),
            "$h = \\{{{}\\}}$".format(", ".join(map(str, self.hidden_sizes)))
        ]
        
        if self.k_max:
            configuration.append("$k_{{\\mathrm{{max}}}} = {}$".format(self.k_max))
        
        if self.count_sum_feature:
            configuration.append("CS")
        
        if self.batch_normalisation:
            configuration.append("BN")
        
        if self.number_of_warm_up_epochs:
            configuration.append("$W = {}$".format(
                self.number_of_warm_up_epochs))
        
        title += " (" + ", ".join(configuration) + ")"
        
        return title
    
    @property
    def description(self):
        
        description_parts = ["Model setup:"]
        
        description_parts.append("type: {}".format(self.type))
        description_parts.append("feature size: {}".format(self.Dim_x))
        description_parts.append("latent size: {}".format(self.Dim_z))
        description_parts.append("hidden sizes: {}".format(", ".join(
            map(str, self.hidden_sizes))))
        
        description_parts.append("latent distribution: " +
            self.latent_distribution_name)
        if "mixture" in self.latent_distribution_name:
            description_parts.append("latent clusters: {}".format(
                self.Dim_y))
        
        description_parts.append("reconstruction distribution: " +
            self.reconstruction_distribution_name)
        if self.k_max > 0:
            description_parts.append(
                "reconstruction classes: {}".format(self.k_max) +
                " (including 0s)"
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
        
        if self.batch_normalisation:
            description_parts.append("using batch normalisation")
        if self.count_sum_feature:
            description_parts.append("using count sums")
        
        description = "\n    ".join(description_parts)
        
        return description
    
    @property
    def parameters(self, trainable = True):
        
        with self.graph.as_default():
            all_parameters = tf.global_variables()
            trainable_parameters = tf.trainable_variables()
        
        if trainable:
            parameters_string_parts = ["Trainable parameters"]
            parameters = trainable_parameters
        elif not trainable:
            parameters_string_parts = ["Non-trainable parameters"]
            parameters = [p for p in all_parameters
                if p not in trainable_parameters]
        
        width = max(map(len, [p.name for p in parameters]))
        
        for parameter in parameters:
            parameters_string_parts.append("{:{}}  {}".format(
                parameter.name, width, parameter.get_shape()))
        
        parameters_string = "\n    ".join(parameters_string_parts)
        
        return parameters_string
    
    def inference(self):
        # Total number of samples drawn from latent distributions, z1 and z2.
        self.S_iw_mc = self.S_iw * self.S_mc
        '''
        ########## ENCODER ###########
        Inference model for:
            q(y,z_1,z_2, x) = q(z_2|y, z_1) q(y|x, z_1) q(z_1|x)
            q(z_1|x)        = N(z_1; mu_{q(z_1)}(x), sigma^2_{q(z_1)}(x)I)
            q(z_2|y,z_1)   = N(z_2; mu_{q(z_2)}(y,z_1), sigma^2_{q(z_2)}(y,z_1)I)
            q(y|x, z_1)     = Cat(y; pi(x, z_1))
        '''
        
        # q(z_1| x) #
        with tf.variable_scope("q_z1"):
            ## (B, H)
            q_z1_NN = dense_layers(
                inputs = self.x,
                num_outputs = self.hidden_sizes,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation,
                is_training = self.is_training,
                dropout_keep_probability = self.dropout_keep_probability,
                scope="NN"
            )

            ## (1, B, L)
            q_z1_mu = tf.reshape(
                dense_layer(
                    q_z1_NN,
                    self.Dim_z,
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                     activation_fn=None,
                    scope="mu"
                    ), 
                [1, -1, self.Dim_z]
            )
            q_z1_log_sigma = tf.reshape(
                dense_layer(
                    q_z1_NN, 
                    self.Dim_z, 
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                     activation_fn=self.log_sigma_support,
                    scope="log_sigma"
                    ), 
                [1, -1, self.Dim_z]
            )

            ## (1, B, L)
            self.q_z1_given_x = Normal(loc=q_z1_mu,
                scale=tf.exp(q_z1_log_sigma),
                validate_args=True
            )

            ## (1, B, L) -> (B, L)
            self.z1_mean = self.q_z1_given_x.mean()

            ## (S_iw * S_mc, B, L)
            self.z1 = tf.reshape(
                self.q_z1_given_x.sample(self.S_iw_mc), 
                [self.S_iw_mc, -1, self.Dim_z]
            )

        # q(y|x,z1)
        with tf.variable_scope("q_y"):
            ## (S_iw * S_mc, B, F)
            x_tile = tf.tile(tf.expand_dims(self.x, 0), [self.S_iw_mc, 1, 1])

            ## (S_iw * S_mc, B, F + L)
            x_z1 = tf.concat((x_tile, self.z1), axis = -1)

            ## (S_iw * S_mc * B, H)
            q_y_NN = dense_layers(
                inputs = x_z1,
                num_outputs = self.hidden_sizes,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation,
                is_training = self.is_training,
                dropout_keep_probability = self.dropout_keep_probability,
                scope="NN"
            )

            ## (S_iw * S_mc, 1, B, K)
            q_y_logits = tf.reshape(dense_layer(q_y_NN, self.Dim_y,
                is_training = self.is_training,
                dropout_keep_probability = self.dropout_keep_probability,
                activation_fn=None,
                scope="logits"), [self.S_iw_mc, 1, -1, self.Dim_y])

            ## (S_iw * S_mc, 1, B, K)
            self.q_y_given_x_z1 = Categorical(
                logits = q_y_logits,
                validate_args=True
            )

            ## (S_iw * S_mc, 1, B, K) -->
            ## (S_iw * S_mc, K, B)
            self.q_y_given_x_z1_probs = tf.reshape(
                tf.transpose(
                    self.q_y_given_x_z1.probs, 
                    [0, 3, 2, 1]
                ),
                [self.S_iw_mc, self.Dim_y, -1]
            )

            ## (B, K) <--  (S_iw * S_mc, K, B)
            self.q_y_mean = tf.transpose(tf.reduce_mean(self.q_y_given_x_z1_probs, 0), [1, 0])

        # q(z_2| y, z_1)
        with tf.variable_scope("q_z2"):
            ## (K, K)
            y_onehot = tf.diag(tf.ones(self.Dim_y))

            ## (K, K) -->
            ## (S_iw * S_mc, K, B, K)
            self.q_y_tile = tf.tile(
                tf.reshape(
                    y_onehot, 
                    [1, self.Dim_y, 1, self.Dim_y]
                ), 
                [self.S_iw_mc, 1, tf.shape(self.x)[0], 1]
            )

            ## (S_iw * S_mc, B, L) -->
            ## (S_iw * S_mc, K, B, L)
            self.z1_tile = tf.tile(tf.expand_dims(self.z1, 1), [1, self.Dim_y, 1, 1])

            ## (S_iw * S_mc, K, B, L + K)
            z1_y = tf.concat((self.z1_tile, self.q_y_tile), axis = -1)

            ## (S_iw * S_mc * K * B, H)
            q_z2_NN = dense_layers(
                inputs = z1_y,
                num_outputs = self.hidden_sizes,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation,
                is_training = self.is_training,
                dropout_keep_probability = self.dropout_keep_probability,
                scope="NN"
            )

            ## (S_iw * S_mc * K * B, L) -->
            ## (1, S_iw * S_mc, K, B, L) 
            q_z2_mu = tf.reshape(
                dense_layer(
                    q_z2_NN,
                    self.Dim_z,
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                     activation_fn=None,
                    scope="mu"
                ),
                [1, self.S_iw_mc, self.Dim_y, -1, self.Dim_z]
            )
            q_z2_log_sigma = tf.reshape(
                dense_layer(
                    q_z2_NN, 
                    self.Dim_z, 
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                     activation_fn=self.log_sigma_support,
                    scope="log_sigma"
                ), 
                [1, self.S_iw_mc, self.Dim_y, -1, self.Dim_z]
            )

            ## (1, S_iw * S_mc, K, B, L) 
            self.q_z2_given_y_z1 = Normal(
                loc=q_z2_mu,
                scale=tf.exp(q_z2_log_sigma),
                validate_args=True
            )

            # TODO: Reduce dim. of mean.
            ##  (1, S_iw * S_mc, K, B, L)  -> (S_iw * S_mc, K, B, L) -->
            ## (S_iw * S_mc, B, L) -->
            ## (B, L)
            self.z2_mean = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reshape(self.q_z2_given_y_z1.mean(), [self.S_iw_mc, self.Dim_y, -1, self.Dim_z]) *\
                        tf.expand_dims(self.q_y_given_x_z1_probs, -1),
                    axis = 1
                ),
            0)
            
            ## (S_iw * S_mc, S_iw * S_mc, K, B, L)
            self.z2 = tf.reshape(
                self.q_z2_given_y_z1.sample(self.S_iw_mc), 
                [self.S_iw_mc, self.S_iw_mc, self.Dim_y, -1, self.Dim_z]
            )

        '''
        ##########DECODER ###########
        Generative model for:
            p(x,y,z_1,z_2) = p(x|y,z_1) p(z_1|y, z_2) p(y|z_2) p(z_2) where:
                p(z_1|y, z_2)   = N(z_1; mu(y,z_2), sigma^2(y,z_2)I)
                p(z_2)          = N(z_2, 0, I)
                p(y|z_2)        = Cat(y; pi(z_2))
                p(x|y, z_1)     = f(x; gamma(y, z_1)) (some count distribution) 
        '''
        # Add a feature which is the total number of counts in an example. 

        with tf.variable_scope("count_sum"):
            if self.count_sum or self.count_sum_feature:
                ## (B, 1) -->
                ## (S_iw * S_mc, S_iw * S_mc, K, B, 1) 
                n_tile = tf.tile(
                    tf.reshape(self.n, [1, 1, 1, -1, 1]
                    ), 
                    [self.S_iw_mc, self.S_iw_mc, self.Dim_y, 1, 1]
                )
        
        # p(z_2) = N(z_2; 0, 1)
        with tf.variable_scope("p_z2"):
            self.p_z2 = Normal(
                loc = 0.,
                scale = 1.,
                validate_args=True
            )


        # p(y| z_2)
        with tf.variable_scope("p_y"):
            if self.count_sum_feature:
                # Add count_sum to z_2 dim.
                ## (S_iw * S_mc, S_iw * S_mc, K, B, L) -->
                ## (S_iw * S_mc, S_iw * S_mc, K, B, L + 1)
                z2_input = tf.concat((self.z2, n_tile), axis = -1, name = "z2_n")
            else: 
                z2_input = self.z2

            ## (S_iw * S_mc * S_iw * S_mc * K * B, H)
            p_y_NN = dense_layers(
                inputs = z2_input,
                num_outputs = self.hidden_sizes,
                reverse_order = True,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation,
                is_training = self.is_training,
                dropout_keep_probability = self.dropout_keep_probability,
                scope="NN"
            )

            ## (S_iw * S_mc, S_iw * S_mc, K, B, K)
            p_y_logits = tf.reshape(
                dense_layer(
                    p_y_NN, self.Dim_y,
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                    activation_fn=None,
                    scope="logits"
                ), 
                [self.S_iw_mc, self.S_iw_mc, 
                    self.Dim_y, -1, self.Dim_y]
            )

            ## (S_iw * S_mc, S_iw * S_mc, K, B, K)
            self.p_y_given_z2 = Categorical(
                logits = p_y_logits,
                validate_args=True
            )


        # p(z_1|z_2, y)
        with tf.variable_scope("p_z1"):
            ## (S_iw * S_mc, K, B, K) -->
            ## (S_iw * S_mc, S_iw * S_mc, K, B, K)
            self.y_tile_p_z1 = tf.tile(
                tf.expand_dims(self.q_y_tile, 0), 
                [self.S_iw_mc, 1, 1, 1, 1], name = "y_tile"
            )
            if self.count_sum_feature:
                # Add count_sum to y and z_2 features concatenated.
                ## (S_iw * S_mc, S_iw * S_mc, K, B, L + K + 1)
                z2_y = tf.concat((self.z2, self.y_tile_p_z1, n_tile), axis = -1, name = "z2_y_n")
            else: 
                # Concat y and z2 features.
                ## (S_iw * S_mc, S_iw * S_mc, K, B, L + K + 1)
                z2_y = tf.concat((self.z2, self.y_tile_p_z1), axis = -1, name = "z2_y")

            ## (S_iw * S_mc * S_iw * S_mc * K * B, H)
            p_z1_NN = dense_layers(
                inputs = z2_y,
                num_outputs = self.hidden_sizes,
                reverse_order = True,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation,
                is_training = self.is_training,
                dropout_keep_probability = self.dropout_keep_probability,
                scope="NN"
            )

            ## (S_iw * S_mc * S_iw * S_mc * K * B, L) -->
            ## (S_iw * S_mc, S_iw * S_mc, K, B, L)
            p_z1_mu = tf.reshape(
                dense_layer(
                    p_z1_NN, self.Dim_z,
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                    activation_fn=None,
                    scope="mu"
                ), 
                [self.S_iw_mc, self.S_iw_mc, self.Dim_y, -1, self.Dim_z]
            )
            p_z1_log_sigma = tf.reshape(
                dense_layer(
                    p_z1_NN, 
                    self.Dim_z, 
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                     activation_fn=self.log_sigma_support,
                    scope="log_sigma"
                ), 
                [self.S_iw_mc, self.S_iw_mc, self.Dim_y, -1, self.Dim_z]
            )

            ## (S_iw * S_mc, S_iw * S_mc, K, B, L)
            self.p_z1_given_y_z2 = Normal(
                loc=p_z1_mu,
                scale=tf.exp(p_z1_log_sigma),
                validate_args=True
            )

            # Gather parameters pi, mu and sigma for p(y,z) = p(z|y)p(y)
            # p(y)
            # (S_iw * S_mc, S_iw * S_mc, K, B, K) --> 
            # (S_iw * S_mc, K, B, K) ---> 
            # (S_iw * S_mc, B, K) -->
            # (K)
            self.p_z_probabilities = tf.reduce_mean(
                tf.reduce_sum(
                    tf.expand_dims(self.q_y_given_x_z1_probs, -1) *\
                    tf.reduce_mean(self.p_y_given_z2.probs, 0),
                    1
                ), 
                (0, 1)
            ) 
            # mu_p(z1|y)
            # (S_iw * S_mc, S_iw * S_mc, K, B, L) --> 
            # (K, L)
            self.p_z_means = tf.reduce_mean(
                self.p_z1_given_y_z2.mean(),
                (0, 1, 3)
            )

            # sigma_p(z1|y)
            # (S_iw * S_mc, S_iw * S_mc, K, B, L) --> 
            # (K, L)
            self.p_z_vars = tf.reduce_mean(
                tf.square(self.p_z1_given_y_z2.stddev()),
                (0, 1, 3)
            )

            self.p_z_vars = tf.clip_by_value(
                self.p_z_vars,
                self.epsilon, 
                self.p_z_vars
            )

# Reconstruction distribution parameterisation
        
        with tf.variable_scope("p_x_given_y_z1"):
            ## (S_iw * S_mc * K * B, H)
            p_x_NN = dense_layers(
            inputs = z1_y,
            num_outputs = self.hidden_sizes,
            reverse_order = True,
            activation_fn = relu,
            batch_normalisation = self.batch_normalisation,
            is_training = self.is_training,
            dropout_keep_probability = self.dropout_keep_probability,
            scope="NN"
            )

            x_theta = {}
            ## (S_iw * S_mc, K, B, F)
            for parameter in self.reconstruction_distribution["parameters"]:
                
                parameter_activation_function = \
                    self.reconstruction_distribution["parameters"]\
                    [parameter]["activation function"]
                p_min, p_max = \
                    self.reconstruction_distribution["parameters"]\
                    [parameter]["support"]
                
                ## (S_iw * S_mc * K * B, H) -->
                ## (S_iw * S_mc * K * B, F) -->
                ## (S_iw * S_mc, K, B, F)
                x_theta[parameter] = tf.reshape(
                    dense_layer(
                        inputs = p_x_NN,
                        num_outputs = self.Dim_x,
                        activation_fn = lambda x: tf.clip_by_value(
                            parameter_activation_function(x),
                            p_min + self.epsilon,
                            p_max - self.epsilon
                        ),
                        is_training = self.is_training,
                        dropout_keep_probability = self.dropout_keep_probability,
                        scope = parameter.upper()
                    ),
                    [self.S_iw_mc, self.Dim_y, -1, self.Dim_x]
                )
            
            if "constrained" in self.reconstruction_distribution_name or \
                "multinomial" in self.reconstruction_distribution_name:
                self.p_x_given_z1_y = self.reconstruction_distribution["class"](
                    x_theta,
                    n_tile[0]
                )
            elif "multinomial" in self.reconstruction_distribution_name:
                self.p_x_given_z1_y = self.reconstruction_distribution["class"](
                    x_theta,
                    n_tile[0]
                )
            else:
                self.p_x_given_z1_y = self.reconstruction_distribution["class"](
                    x_theta
                )
            
            if self.k_max:
                x_logits = dense_layer(
                    inputs = p_x_NN,
                    num_outputs = self.Dim_x * self.k_max,
                    activation_fn = None,
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability,
                    scope = "P_K"
                )
                
                x_logits = tf.reshape(x_logits,
                    [self.S_iw_mc, self.Dim_y, -1, self.Dim_x, self.k_max])
                
                self.p_x_given_z1_y = Categorized(
                    dist = self.p_x_given_z1_y,
                    cat = Categorical(logits = x_logits, validate_args=True)
                )
            
            ## (S_iw * S_mc, K, B, F) -->
            ## (S_iw * S_mc, B, F) -->
            ## (B, F)
            self.x_mean = tf.reduce_mean(tf.reduce_sum(self.p_x_given_z1_y.mean() * tf.expand_dims(self.q_y_given_x_z1_probs, -1), axis = 1), 0)
        
        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            parameter_summary = tf.summary.histogram(parameter.name, parameter)
            self.parameter_summary_list.append(parameter_summary)
        self.parameter_summary = tf.summary.merge(self.parameter_summary_list)
    
    def loss(self):
        # Loss
        # Initialise reshaped data
        ## (S_iw * S_mc, K, B, F)
        t_tiled = tf.tile(
            tf.reshape(
                self.t, 
                [1, 1, -1, self.Dim_x]
            ),
            [self.S_iw_mc, self.Dim_y, 1, 1]
        )

        # log(p(x|y,z1))
        ## (S_iw * S_mc, K, B, F) -->
        ## (S_iw * S_mc, K, B) -->
        p_x_given_z1_y_log_prob = tf.reduce_sum(
            self.p_x_given_z1_y.log_prob(t_tiled), -1
        )
        ## (S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, B) -->
        ## (B)
        log_p_x_given_z1_y = tf.reduce_mean(
            tf.reduce_sum(
                self.q_y_given_x_z1_probs * p_x_given_z1_y_log_prob, 
                1
            ),
            0
        )

        # log(p(z_1|z_2, y))
        ## (S_iw * S_mc, S_iw * S_mc, K, B, L) -->
        ## (S_iw * S_mc, S_iw * S_mc, K, B)
        p_z1_given_y_z2_log_prob = tf.reduce_sum(
            self.p_z1_given_y_z2.log_prob(
                tf.tile(tf.expand_dims(self.z1_tile, 0), 
                    [self.S_iw_mc, 1, 1, 1, 1]
                )
            ),
            -1
        )
        ## (S_iw * S_mc, S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, B) -->
        ## (B)
        log_p_z1_given_y_z2 = tf.reduce_mean(
            tf.reduce_sum(
                self.q_y_given_x_z1_probs * tf.reduce_mean(
                    p_z1_given_y_z2_log_prob, 
                    0
                ), 
                1
            ),
            0
        )


        # log(p(y|z_2))
        ## (S_iw * S_mc, S_iw * S_mc, K, B, K) -->
        ## (S_iw * S_mc, S_iw * S_mc, K, B)
        p_y_given_z2_log_prob = self.p_y_given_z2.log_prob(
            tf.argmax(self.y_tile_p_z1, axis = -1)
        )
        ## (S_iw * S_mc, S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, B) -->
        ## (B)
        log_p_y_given_z2 = tf.reduce_mean(
            tf.reduce_sum(
                self.q_y_given_x_z1_probs * tf.reduce_mean(
                    p_y_given_z2_log_prob,
                    0
                ), 
                1
            ),
            0
        )


        # log(p(z_2))
        ## (S_iw * S_mc, S_iw * S_mc, K, B, L) -->
        ## (S_iw * S_mc, S_iw * S_mc, K, B)
        p_z2_log_prob = tf.reduce_sum(self.p_z2.log_prob(self.z2), -1)

        ## (S_iw * S_mc, S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, B) -->
        ## (B)
        log_p_z2 = tf.reduce_mean(
            tf.reduce_sum(
                self.q_y_given_x_z1_probs * tf.reduce_mean(
                    p_z2_log_prob, 
                    0
                ), 
                1
            ),
            0
        )

        # log(q(z_2|y, z_1))
        ## (S_iw * S_mc, S_iw * S_mc, K, B, L) -->
        ## (S_iw * S_mc, S_iw * S_mc, K, B)
        q_z2_given_y_z1_log_prob = tf.reduce_sum(
            self.q_z2_given_y_z1.log_prob(self.z2),
            -1
        )
        ## (S_iw * S_mc, S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, B) -->
        ## (B)
        log_q_z2_given_y_z1 = tf.reduce_mean(
            tf.reduce_sum(
                self.q_y_given_x_z1_probs * tf.reduce_mean(
                    q_z2_given_y_z1_log_prob, 
                    0
                ), 
                1
            ),
            0
        )

        # log(q(y|z_1, x))
        ## (S_iw * S_mc, K, B, K) -->
        ## (S_iw * S_mc, K, B)
        q_y_given_x_z1_log_prob = self.q_y_given_x_z1.log_prob(
            tf.argmax(self.q_y_tile, axis = -1)
        )
        ## (S_iw * S_mc, K, B) -->
        ## (S_iw * S_mc, B) -->
        ## (B)
        log_q_y_given_x_z1 = tf.reduce_mean(
            tf.reduce_sum(
                self.q_y_given_x_z1_probs * q_y_given_x_z1_log_prob, 
                1
            ),
            0
        )

        # log(q(z_1|x))
        ## (S_iw * S_mc, B, L) -->
        ## (S_iw * S_mc, B)
        q_z1_given_x_log_prob = tf.reduce_sum(
            self.q_z1_given_x.log_prob(self.z1),
            -1
        )

        ## (S_iw * S_mc, B) -->
        ## (B)
        log_q_z1_given_x = tf.reduce_mean(
            q_z1_given_x_log_prob, 
            0
        )

        # Importance weighted log likelihood
        # Put all log_prob tensors together in one.
        ## (S_iw * S_mc, S_iw * S_mc, K, B) -->
        ## (S_iw, S_mc, S_iw, S_mc, K, B)
        # all_log_prob_iw = tf.reshape(
        #     tf.expand_dims(p_x_given_z1_y_log_prob, 0)\
        #     - self.warm_up_weight * (q_z2_given_y_z1_log_prob \
        #         + tf.expand_dims(q_y_given_x_z1_log_prob, 0) \
        #         + tf.expand_dims(tf.expand_dims(q_z1_given_x_log_prob, 1), 0) \
        #         - p_z2_log_prob \
        #         - p_z1_given_y_z2_log_prob \
        #         - p_y_given_z2_log_prob
        #         ), 
        #     [self.S_iw, self.S_mc, self.S_iw, self.S_mc, self.Dim_y, -1]
        # )

        # # log-mean-exp trick for stable marginalisation of importance weights.
        # ## (S_iw, S_mc, S_iw, S_mc, K, B) -->
        # ## (S_mc, S_mc, K, B)
        # log_mean_exp_iw = log_reduce_exp(
        #     all_log_prob_iw, 
        #     reduction_function=tf.reduce_mean, axis = (0, 2)
        # )

        # # Marginalise all Monte Carlo samples, classes and examples into total
        # # importance weighted loss
        # # (S_iw * S_mc, K, B) --> (S_iw, S_mc, K, B) --> (S_mc, K, B)
        # q_y_given_x_z1_probs_mc = tf.reshape(
        #     self.q_y_given_x_z1_probs, 
        #     [self.S_iw, self.S_mc, self.Dim_y, -1]
        # )[0]

        # ## (S_mc, S_mc, K, B) -->
        # ## (S_mc, K, B) -->
        # ## (S_mc, B) -->
        # ## ()
        # self.ELBO = tf.reduce_mean(
        #     tf.reduce_sum(
        #         q_y_given_x_z1_probs_mc * tf.reduce_mean(
        #             log_mean_exp_iw, 
        #             0
        #         ), 
        #         1
        #     ),
        # )
        # self.loss = self.ELBO
        # tf.add_to_collection('losses', self.ELBO)

        # KL_z1
        self.KL_z1 = tf.reduce_mean(log_q_z1_given_x - log_p_z1_given_y_z2)

        # KL_z2
        self.KL_z2 = tf.reduce_mean(log_q_z2_given_y_z1 - log_p_z2)

        # KL_y
        self.KL_y = tf.reduce_mean(log_q_y_given_x_z1 - log_p_y_given_z2)

        self.KL = tf.add_n([self.KL_z1, self.KL_z2, self.KL_y], name = 'KL')
        tf.add_to_collection('losses', self.KL)

        self.KL_all = tf.expand_dims(self.KL, -1, name = "KL_all")

        # ENRE
        self.ENRE = tf.reduce_mean(log_p_x_given_z1_y, name = "ENRE")
        tf.add_to_collection('losses', self.ENRE)

        # ELBO
        self.ELBO = tf.subtract(self.ENRE, self.KL, name = "ELBO")
        tf.add_to_collection('losses', self.ELBO)

        # loss objective with Warm-up and term-specific KL weighting 
        self.loss = self.ENRE - self.warm_up_weight * (
            self.KL_z1 + self.KL_z2 + self.KL_y
        )
    
    def training(self):
        
        # Create the gradient descent optimiser with the given learning rate.
        def setupTraining():
            
            # Optimizer and training objective of negative loss
            optimiser = tf.train.AdamOptimizer(self.learning_rate)
            # clipped_optimiser = tf.contrib.opt.VariableClippingOptimizer(optimiser, ) 
            # Create a variable to track the global step.
            self.global_step = tf.Variable(0, name = 'global_step',
                trainable = False)
            
            # Use the optimiser to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training
            # step.
            # self.train_op = optimiser.minimize(
            #     -self.loss,
            #     global_step = self.global_step
            # )
        
            gradients = optimiser.compute_gradients(-self.loss)
            # for gradient, variable in gradients:
            #     if not gradient.:
            #         print(variable)
            clipped_gradients = []
            for gradient, variable in gradients:
                if gradient is not None:
                    clipped_gradients.append((tf.clip_by_value(gradient, -1., 1.), variable))
                else:
                    clipped_gradients.append((gradient, variable))
            # clipped_gradients = [(tf.clip_by_value(gradient, -1., 1.), variable) for gradient, variable in gradients if gradient is not None else (gradient, variable)]
            self.train_op = optimiser.apply_gradients(clipped_gradients, global_step = self.global_step)
        # Make sure that the updates of the moving_averages in batch_norm
        # layers are performed before the train_step.
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                setupTraining()
        else:
            setupTraining()

    def train(self, training_set, validation_set,
        number_of_epochs = 100, batch_size = 100, learning_rate = 1e-3,
        reset_training = False):
        
        # Logging
        
        status = {
            "completed": False,
            "message": None,
            "trained": None,
            "training time": None,
            "last epoch time": None
        }
        
        # parameter_values = "lr_{:.1g}".format(learning_rate)
        # parameter_values += "_b_" + str(batch_size)
        
        # self.log_directory = os.path.join(self.log_directory, parameter_values)
        
        if reset_training and os.path.exists(self.log_directory):
            shutil.rmtree(self.log_directory)
        
        checkpoint_file = os.path.join(self.log_directory, 'model.ckpt')
        
        # Setup
        
        batch_size /= self.number_of_importance_samples["training"] \
            * self.number_of_monte_carlo_samples["training"]
        batch_size = int(numpy.ceil(batch_size))
        
        if self.count_sum:
            n_train = training_set.normalised_count_sum
            n_valid = validation_set.normalised_count_sum
        
        M_train = training_set.number_of_examples
        M_valid = validation_set.number_of_examples
        
        noisy_preprocess = training_set.noisy_preprocess
        
        if not noisy_preprocess:
            
            x_train = training_set.preprocessed_values
            x_valid = validation_set.preprocessed_values
        
            if self.reconstruction_distribution_name == "bernoulli":
                t_train = training_set.binarised_values
                t_valid = validation_set.binarised_values
            else:
                t_train = training_set.values
                t_valid = validation_set.values
        
        steps_per_epoch = numpy.ceil(M_train / batch_size)
        output_at_step = numpy.round(numpy.linspace(0, steps_per_epoch, 11))
        
        with tf.Session(graph = self.graph) as session:
            
            parameter_summary_writer = tf.summary.FileWriter(
                self.log_directory)
            training_summary_writer = tf.summary.FileWriter(
                os.path.join(self.log_directory, "training"))
            validation_summary_writer = tf.summary.FileWriter(
                os.path.join(self.log_directory, "validation"))
            
            # Initialisation
            
            checkpoint = tf.train.get_checkpoint_state(self.log_directory)
            
            if checkpoint:
                self.saver.restore(session, checkpoint.model_checkpoint_path)
                epoch_start = int(os.path.split(
                    checkpoint.model_checkpoint_path)[-1].split('-')[-1])
            else:
                session.run(tf.global_variables_initializer())
                epoch_start = 0
                parameter_summary_writer.add_graph(session.graph)
            
            status["trained"] = "{}-{}".format(epoch_start, number_of_epochs)
            
            # Training loop
            
            data_string = dataString(training_set,
                self.reconstruction_distribution_name)
            print(trainingString(epoch_start, number_of_epochs, data_string))
            print()
            training_time_start = time()
            
            for epoch in range(epoch_start, number_of_epochs):
                
                epoch_time_start = time()
                
                if noisy_preprocess:
                    x_train = noisy_preprocess(training_set.values)
                    t_train = x_train
                    x_valid = noisy_preprocess(validation_set.values)
                    t_valid = x_valid

                if self.number_of_warm_up_epochs:
                    warm_up_weight = float(min(
                        epoch / (self.number_of_warm_up_epochs), 1.0))
                else:
                    warm_up_weight = 1.0
                
                shuffled_indices = numpy.random.permutation(M_train)
                
                for i in range(0, M_train, batch_size):
                    
                    # Internal setup
                    
                    step_time_start = time()
                    
                    step = session.run(self.global_step)
                    
                    # Prepare batch
                    
                    batch_indices = shuffled_indices[i:(i + batch_size)]
                    
                    feed_dict_batch = {
                        self.x: x_train[batch_indices],
                        self.t: t_train[batch_indices],
                        self.is_training: True,
                        self.learning_rate: learning_rate, 
                        self.warm_up_weight: warm_up_weight,
                        self.S_iw: self.number_of_importance_samples["training"],
                        self.S_mc: self.number_of_monte_carlo_samples["training"]
                    }
                    
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[batch_indices]
                    
                    # Run the stochastic batch training operation
                    _, batch_loss = session.run(
                        [self.train_op, self.ELBO],
                        feed_dict = feed_dict_batch
                    )
                    
                    # Compute step duration
                    step_duration = time() - step_time_start
                    
                    # Print evaluation and output summaries
                    if (step + 1 - steps_per_epoch * epoch) in output_at_step:
                        
                        print('Step {:d} ({}): {:.5g}.'.format(
                            int(step + 1), formatDuration(step_duration),
                            batch_loss))
                        
                        if numpy.isnan(batch_loss):
                            status["completed"] = False
                            status["message"] = "loss became nan"
                            status["training time"] = formatDuration(
                                training_time_start - time())
                            status["last epoch time"] = formatDuration(
                                epoch_duration)
                            return status
                
                print()
                
                epoch_duration = time() - epoch_time_start
                
                print("Epoch {} ({}):".format(epoch + 1,
                    formatDuration(epoch_duration)))

                # With warmup or not
                if warm_up_weight < 1:
                    print('    Warm-up weight: {:.2g}'.format(warm_up_weight))

                # Saving model parameters
                print('    Saving model.')
                saving_time_start = time()
                self.saver.save(session, checkpoint_file,
                    global_step = epoch + 1)
                saving_duration = time() - saving_time_start
                print('    Model saved ({}).'.format(
                    formatDuration(saving_duration)))
                
                # Export parameter summaries
                parameter_summary_string = session.run(
                    self.parameter_summary,
                    feed_dict = {self.warm_up_weight: warm_up_weight}
                )
                parameter_summary_writer.add_summary(
                    parameter_summary_string, global_step = epoch + 1)
                parameter_summary_writer.flush()
                
                # Evaluation
                print('    Evaluating model.')
                
                ## Training
                
                evaluating_time_start = time()
                
                ELBO_train = 0
                ENRE_train = 0
                KL_z1_train = 0
                KL_z2_train = 0
                KL_y_train = 0
                
                
                for i in range(0, M_train, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_train[subset]
                    t_batch = t_train[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.warm_up_weight: 1.0,
                        self.S_iw: self.number_of_importance_samples["training"],
                        self.S_mc: self.number_of_monte_carlo_samples["training"]
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[subset]
                    
                    ELBO_i, ENRE_i, KL_z1_i, KL_z2_i, KL_y_i = session.run(
                        [self.ELBO, self.ENRE, self.KL_z1, self.KL_z2, self.KL_y],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_train += ELBO_i
                    ENRE_train += ENRE_i
                    KL_z1_train += KL_z1_i
                    KL_z2_train += KL_z2_i
                    KL_y_train += KL_y_i
                                    
                ELBO_train /= M_train / batch_size
                ENRE_train /= M_train / batch_size
                KL_z1_train /= M_train / batch_size
                KL_z2_train /= M_train / batch_size
                KL_y_train /= M_train / batch_size
                                
                evaluating_duration = time() - evaluating_time_start
                
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_train)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_train)
                summary.value.add(tag="losses/kl_divergence_z1",
                    simple_value = KL_z1_train)                
                summary.value.add(tag="losses/kl_divergence_z2",
                    simple_value = KL_z2_train)
                summary.value.add(tag="losses/kl_divergence_y",
                    simple_value = KL_y_train)
                
                training_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                training_summary_writer.flush()
                
                print("    Training set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL_z1: {:.5g}, KL_z2: {:.5g}, KL_y: {:.5g}.".format(
                    ELBO_train, ENRE_train, KL_z1_train, KL_z2_train, KL_y_train))
                
                ## Validation
                
                evaluating_time_start = time()
                
                ELBO_valid = 0
                ENRE_valid = 0
                KL_z1_valid = 0
                KL_z2_valid = 0
                KL_y_valid = 0
                p_z_probabilities = numpy.zeros(self.Dim_y)
                p_z_means = numpy.zeros((self.Dim_y, self.Dim_z))
                p_z_vars = numpy.zeros((self.Dim_y, self.Dim_z))
                z_mean_valid = numpy.empty([M_valid, self.Dim_z])

                for i in range(0, M_valid, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_valid[subset]
                    t_batch = t_valid[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.warm_up_weight: 1.0,
                        self.S_iw:
                            self.number_of_importance_samples["training"],
                        self.S_mc:
                            self.number_of_monte_carlo_samples["training"]
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_valid[subset]
                    
                    ELBO_i, ENRE_i, KL_z1_i, KL_z2_i, KL_y_i, p_z_probabilities_i,\
                    p_z_means_i, p_z_vars_i, z_mean_i = session.run(
                        [self.ELBO, self.ENRE, self.KL_z1, self.KL_z2, self.KL_y, self.p_z_probabilities, self.p_z_means, self.p_z_vars, self.z1_mean],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_valid += ELBO_i
                    ENRE_valid += ENRE_i
                    KL_z1_valid += KL_z1_i
                    KL_z2_valid += KL_z2_i
                    KL_y_valid += KL_y_i
                    p_z_probabilities += p_z_probabilities_i
                    p_z_means += p_z_means_i
                    p_z_vars += p_z_vars_i
                    z_mean_valid[subset] = z_mean_i

                ELBO_valid /= M_valid / batch_size
                ENRE_valid /= M_valid / batch_size
                KL_z1_valid /= M_valid / batch_size
                KL_z2_valid /= M_valid / batch_size
                KL_y_valid /= M_valid / batch_size
                p_z_probabilities /= M_valid / batch_size
                p_z_means /= M_valid / batch_size
                p_z_vars /= M_valid / batch_size


                evaluating_duration = time() - evaluating_time_start
                
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_valid)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_valid)
                summary.value.add(tag="losses/kl_divergence_z1",
                    simple_value = KL_z1_valid)                
                summary.value.add(tag="losses/kl_divergence_z2",
                    simple_value = KL_z2_valid)
                summary.value.add(tag="losses/kl_divergence_y",
                    simple_value = KL_y_valid)
                
                ### Centroids
                for k in range(self.Dim_y):
                    summary.value.add(
                        tag="prior/cluster_{}/probability".format(k),
                        simple_value = p_z_probabilities[k]
                    )
                    for l in range(self.Dim_z):
                        summary.value.add(
                            tag="prior/cluster_{}/mean/dimension_{}".format(k, l),
                            simple_value = p_z_means[k, l]
                        )
                        summary.value.add(
                            tag="prior/cluster_{}/variance/dimension_{}"\
                                .format(k, l),
                            simple_value = p_z_vars[k, l]
                        )

                validation_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                validation_summary_writer.flush()
                
                print("    Validation set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL_z1: {:.5g}, KL_z2: {:.5g}, KL_y: {:.5g}.".format(
                    ELBO_valid, ENRE_valid, KL_z1_valid, KL_z2_valid, KL_y_valid))
                
                print()

                # Plot latent validation values
                under_10 = epoch < 10
                under_100 = epoch < 100 and (epoch + 1) % 10 == 0
                under_1000 = epoch < 1000 and (epoch + 1) % 50 == 0 
                last_one = epoch == number_of_epochs - 1
                if under_10 or under_100 or under_1000 or last_one:
                    if "mixture" in self.latent_distribution_name:
                        K = self.Dim_y
                        L = self.Dim_z
                        p_z_covariance_matrices = numpy.empty([K, L, L])
                        for k in range(K):
                            p_z_covariance_matrices[k] = numpy.diag(p_z_vars[k])
                        centroids = {
                            "prior": {
                                "probabilities": p_z_probabilities,
                                "means": p_z_means,
                                "covariance_matrices": p_z_covariance_matrices
                            }
                        }
                    else:
                        centroids = None
                    analyseIntermediateResults(
                        z_mean_valid, validation_set, centroids, epoch,
                        self.training_name, self.main_results_directory
                    )
                    print()
            training_duration = time() - training_time_start
            
            if epoch_start >= number_of_epochs:
                epoch_duration = training_duration
            else:
                print("Model trained for {} epochs ({}).".format(
                    number_of_epochs, formatDuration(training_duration)))
            
            # Clean up
            
            checkpoint = tf.train.get_checkpoint_state(self.log_directory)
            
            if checkpoint:
                for f in os.listdir(self.log_directory):
                    file_path = os.path.join(self.log_directory, f)
                    is_old_checkpoint_file = os.path.isfile(file_path) \
                        and "model" in f \
                        and not checkpoint.model_checkpoint_path in file_path
                    if is_old_checkpoint_file:
                        os.remove(file_path)
            
            status["completed"] = True
            status["training time"] = formatDuration(training_duration)
            status["last epoch time"] = formatDuration(epoch_duration)
            
            return status
    
    def evaluate(self, test_set, batch_size = 100):
        
        batch_size /= self.number_of_importance_samples["evaluation"] \
            * self.number_of_monte_carlo_samples["evaluation"]
        batch_size = int(numpy.ceil(batch_size))
        
        if self.count_sum:
            n_test = test_set.normalised_count_sum
        
        M_test = test_set.number_of_examples
        F_test = test_set.number_of_features
        
        noisy_preprocess = test_set.noisy_preprocess
        
        if not noisy_preprocess:
            
            x_test = test_set.preprocessed_values
        
            if self.reconstruction_distribution_name == "bernoulli":
                t_test = test_set.binarised_values
            else:
                t_test = test_set.values
            
        else:
            x_test = noisy_preprocess(test_set.values)
            t_test = x_test
        
        checkpoint = tf.train.get_checkpoint_state(self.log_directory)
        
        test_summary_directory = os.path.join(self.log_directory, "test")
        if os.path.exists(test_summary_directory):
            shutil.rmtree(test_summary_directory)
        
        with tf.Session(graph = self.graph) as session:
            
            test_summary_writer = tf.summary.FileWriter(
                test_summary_directory)
            
            if checkpoint:
                self.saver.restore(session, checkpoint.model_checkpoint_path)
                epoch = int(os.path.split(
                    checkpoint.model_checkpoint_path)[-1].split('-')[-1])
            else:
                raise Exception(
                    "Cannot evaluate model when it has not been trained.")
            
            data_string = dataString(test_set,
                self.reconstruction_distribution_name)
            print('Evaluating trained model on {}.'.format(data_string))
            evaluating_time_start = time()
            
            ELBO_test = 0
            ENRE_test = 0
            KL_z1_test = 0
            KL_z2_test = 0
            KL_y_test = 0
            
            x_mean_test = numpy.empty([M_test, F_test])
            z1_mean_test = numpy.empty([M_test, self.Dim_z])
            z2_mean_test = numpy.empty([M_test, self.Dim_z])
            y_mean_test = numpy.empty([M_test, self.Dim_y])
            p_z_probabilities = numpy.zeros(self.Dim_y)
            p_z_means = numpy.zeros((self.Dim_y, self.Dim_z))
            p_z_vars = numpy.zeros((self.Dim_y, self.Dim_z))

            for i in range(0, M_test, batch_size):
                subset = slice(i, (i + batch_size))
                x_batch = x_test[subset]
                t_batch = t_test[subset]
                feed_dict_batch = {
                    self.x: x_batch,
                    self.t: t_batch,
                    self.is_training: False,
                    self.warm_up_weight: 1.0,
                    self.S_iw: self.number_of_importance_samples["evaluation"],
                    self.S_mc: self.number_of_monte_carlo_samples["evaluation"]
                }
                if self.count_sum:
                    feed_dict_batch[self.n] = n_test[subset]
                
                ELBO_i, ENRE_i, KL_z1_i, KL_z2_i, KL_y_i, \
                    p_z_probabilities_i, p_z_means_i, p_z_vars_i, \
                    x_mean_i, z1_mean_i, z2_mean_i, y_mean_i = session.run(
                    [self.ELBO, self.ENRE, self.KL_z1, self.KL_z2, self.KL_y,
                        self.p_z_probabilities, self.p_z_means, 
                        self.p_z_vars, self.x_mean, self.z1_mean, 
                        self.z2_mean, self.q_y_mean],
                    feed_dict = feed_dict_batch
                )
                
                ELBO_test += ELBO_i
                ENRE_test += ENRE_i
                KL_z1_test += KL_z1_i
                KL_z2_test += KL_z2_i
                KL_y_test += KL_y_i
                p_z_probabilities += p_z_probabilities_i
                p_z_means += p_z_means_i
                p_z_vars += p_z_vars_i
                
                x_mean_test[subset] = x_mean_i
                z1_mean_test[subset] = z1_mean_i
                z2_mean_test[subset] = z2_mean_i
                y_mean_test[subset] = y_mean_i
            
            ELBO_test /= M_test / batch_size
            ENRE_test /= M_test / batch_size
            KL_z1_test /= M_test / batch_size
            KL_z2_test /= M_test / batch_size
            KL_y_test /= M_test / batch_size
            p_z_probabilities /= M_test / batch_size
            p_z_means /= M_test / batch_size
            p_z_vars /= M_test / batch_size
            
            summary = tf.Summary()
            summary.value.add(tag="losses/lower_bound",
                simple_value = ELBO_test)
            summary.value.add(tag="losses/reconstruction_error",
                simple_value = ENRE_test)
            summary.value.add(tag="losses/kl_divergence_z1",
                simple_value = KL_z1_test)                
            summary.value.add(tag="losses/kl_divergence_z2",
                simple_value = KL_z2_test)
            summary.value.add(tag="losses/kl_divergence_y",
                simple_value = KL_y_test)
            
            ### Centroids
            for k in range(self.Dim_y):
                summary.value.add(
                    tag="prior/cluster_{}/probability".format(k),
                    simple_value = p_z_probabilities[k]
                )
                for l in range(self.Dim_z):
                    summary.value.add(
                        tag="prior/cluster_{}/mean/dimension_{}".format(k, l),
                        simple_value = p_z_means[k, l]
                    )
                    summary.value.add(
                        tag="prior/cluster_{}/variance/dimension_{}"\
                            .format(k, l),
                        simple_value = p_z_vars[k, l]
                    )            

            test_summary_writer.add_summary(summary,
                global_step = epoch + 1)
            test_summary_writer.flush()
            
            evaluating_duration = time() - evaluating_time_start
            print("    Test set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL_z1: {:.5g}, KL_z2: {:.5g}, KL_y: {:.5g}.".format(
                    ELBO_test, ENRE_test, KL_z1_test, KL_z2_test, KL_y_test))
            
            if self.reconstruction_distribution_name == "bernoulli":
                transformed_test_set = DataSet(
                    name = test_set.name,
                    values = t_test,
                    preprocessed_values = None,
                    labels = test_set.labels,
                    example_names = test_set.example_names,
                    feature_names = test_set.feature_names,
                    feature_selection = test_set.feature_selection,
                    preprocessing_methods = test_set.preprocessing_methods,
                    kind = "test",
                    version = "binarised"
                )
            else:
                transformed_test_set = test_set
            
            reconstructed_test_set = DataSet(
                name = test_set.name,
                values = x_mean_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = test_set.feature_names,
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "reconstructed"
            )
            
            z1_test_set = DataSet(
                name = test_set.name,
                values = z1_mean_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = numpy.array(["z_1 variable {}".format(
                    i + 1) for i in range(self.Dim_z)]),
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "z1"
            )
            
            z2_test_set = DataSet(
                name = test_set.name,
                values = z2_mean_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = numpy.array(["z_2 variable {}".format(
                    i + 1) for i in range(self.Dim_z)]),
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "z2"
            )

            y_test_set = DataSet(
                name = test_set.name,
                values = y_mean_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = numpy.array(["y variable {}".format(
                    i + 1) for i in range(self.Dim_y)]),
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "y"
            )
            latent_test_sets = (z1_test_set, z2_test_set, y_test_set)

            return transformed_test_set, reconstructed_test_set, latent_test_sets
