import tensorflow as tf

from models.auxiliary import (
    dense_layer, dense_layers, log_reduce_exp, reduce_logmeanexp,
    epochsWithNoImprovement,
    trainingString, dataString,
    copyModelDirectory, removeOldCheckpoints
)

from tensorflow.python.ops.nn import relu, softmax
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
from auxiliary import loadLearningCurves

class VariationalAutoencoder(object):
    def __init__(self, feature_size, latent_size, hidden_sizes,
        number_of_monte_carlo_samples, number_of_importance_samples,
        analytical_kl_term = False,
        latent_distribution = "gaussian", number_of_latent_clusters = 1,
        parameterise_latent_posterior = False,
        reconstruction_distribution = None, 
        number_of_reconstruction_classes = None, 
        batch_normalisation = True, 
        dropout_keep_probabilities = [],
        count_sum = True,
        number_of_warm_up_epochs = 0, 
        epsilon = 1e-6,
        log_directory = "log", results_directory = "results"):
        
        # Class setup
        super(VariationalAutoencoder, self).__init__()
        
        self.type = "VAE"
        
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        self.latent_distribution_name = latent_distribution
        self.latent_distribution = copy.deepcopy(
            latent_distributions[latent_distribution]
        )
        self.number_of_latent_clusters = number_of_latent_clusters
        self.analytical_kl_term = analytical_kl_term
        self.parameterise_latent_posterior = parameterise_latent_posterior
        
        # Dictionary holding number of samples needed for the "monte carlo" 
        # estimator and "importance weighting" during both "train" and "test" time.  
        self.number_of_importance_samples = number_of_importance_samples
        self.number_of_monte_carlo_samples = number_of_monte_carlo_samples

        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = distributions\
            [reconstruction_distribution]
        
        # Number of categorical elements needed for reconstruction, e.g. K+1
        self.number_of_reconstruction_classes = number_of_reconstruction_classes + 1
        # K: For the sum over K-1 Categorical probabilities and the last K
        #   count distribution pdf.
        self.k_max = number_of_reconstruction_classes
        
        self.batch_normalisation = batch_normalisation
        
        # Dropout keep probabilities (p) for 3 different kinds of layers
        # Hidden layers
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

        self.count_sum_feature = count_sum
        self.count_sum = "constrained" in \
            self.reconstruction_distribution_name or "multinomial" in \
            self.reconstruction_distribution_name

        self.number_of_warm_up_epochs = number_of_warm_up_epochs

        self.epsilon = epsilon
        
        self.main_log_directory = log_directory
        self.main_results_directory = results_directory
        # Early stopping
        
        self.early_stopping_rounds = 10
        self.stopped_early = None
        
        # Graph setup
        
        self.graph = tf.Graph()
        
        self.parameter_summary_list = []
        
        with self.graph.as_default():
            
            self.x = tf.placeholder(tf.float32, [None, self.feature_size], 'X')
            self.t = tf.placeholder(tf.float32, [None, self.feature_size], 'T')
            
            if self.count_sum_feature:
                self.n_feature = tf.placeholder(tf.float32, [None, 1], 'count_sum_feature')

            if self.count_sum:
                self.n = tf.placeholder(tf.float32, [None, 1], 'count_sum')
            
            # self.max_count = tf.placeholder(tf.int32, [1], 'max_count')

            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            
            self.warm_up_weight = tf.placeholder(tf.float32, [], 'warm_up_weight')
            warm_up_weight_summary = tf.summary.scalar('warm_up_weight',
                self.warm_up_weight)
            self.parameter_summary_list.append(warm_up_weight_summary)
            
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')
            self.use_deterministic_z = tf.placeholder(tf.bool, [], 'use_deterministic_z')
            self.number_of_iw_samples = tf.placeholder(
                tf.int32,
                [],
                'number_of_iw_samples'
            )
            self.number_of_mc_samples = tf.placeholder(
                tf.int32,
                [],
                'number_of_mc_samples'
            )

            self.inference()
            self.loss()
            self.training()
            
            self.saver = tf.train.Saver(max_to_keep = 1)
    
    @property
    def name(self):
        
        # Latent parts
        
        latent_parts = [normaliseString(self.latent_distribution_name)]
        
        if self.parameterise_latent_posterior:
            latent_parts.append("parameterised")
        
        if "mixture" in self.latent_distribution_name:
            latent_parts.append("c_{}".format(self.number_of_latent_clusters))
        
        # Reconstruction parts
        
        reconstruction_parts = [normaliseString(
            self.reconstruction_distribution_name)]
        
        if self.k_max:
            reconstruction_parts.append("k_{}".format(self.k_max))
        
        if self.count_sum_feature:
            reconstruction_parts.append("sum")
        
        reconstruction_parts.append("l_{}".format(self.latent_size))
        reconstruction_parts.append(
            "h_" + "_".join(map(str, self.hidden_sizes)))
        
        reconstruction_parts.append("mc_{}".format(
            self.number_of_monte_carlo_samples["training"]))
        reconstruction_parts.append("mc_{}".format(
            self.number_of_importance_samples["training"]))
        
        if self.analytical_kl_term:
            reconstruction_parts.append("kl")
        
        if self.batch_normalisation:
            reconstruction_parts.append("bn")

        if len(self.dropout_parts) > 0:
            reconstruction_parts.append(
                "dropout_" + "_".join(self.dropout_parts))
        
        if self.number_of_warm_up_epochs:
            reconstruction_parts.append("wu_{}".format(
                self.number_of_warm_up_epochs))
        
        # Complete name
        
        latent_part = "-".join(latent_parts)
        reconstruction_part = "-".join(reconstruction_parts)
        
        model_name = os.path.join(self.type, latent_part, reconstruction_part)
        
        return model_name
    
    @property
    def log_directory(self):
        return os.path.join(self.main_log_directory, self.name)
    
    @property
    def early_stopping_log_directory(self):
        return os.path.join(self.log_directory, "early_stopping")
    
    @property
    def best_model_log_directory(self):
        return os.path.join(self.log_directory, "best")
    
    @property
    def title(self):
        
        title = model.type
        
        configuration = [
            self.reconstruction_distribution_name.capitalize(),
            "$l = {}$".format(self.latent_size),
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
        description_parts.append("feature size: {}".format(self.feature_size))
        description_parts.append("latent size: {}".format(self.latent_size))
        description_parts.append("hidden sizes: {}".format(", ".join(
            map(str, self.hidden_sizes))))
        
        description_parts.append("latent distribution: " +
            self.latent_distribution_name)
        if "mixture" in self.latent_distribution_name:
            description_parts.append("latent clusters: {}".format(
                self.number_of_latent_clusters))
        if self.parameterise_latent_posterior:
            description_parts.append("using parameterisation of latent posterior parameters")
        
        description_parts.append("reconstruction distribution: " +
            self.reconstruction_distribution_name)
        if self.k_max > 0:
            description_parts.append(
                "reconstruction classes: {}".format(self.k_max) +
                " (including 0s)"
            )
        
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
        
        if self.analytical_kl_term:
            description_parts.append("using analytical KL term")
        
        if self.batch_normalisation:
            description_parts.append("using batch normalisation")

        if len(self.dropout_parts) > 0:
            description_parts.append("dropout keep probability: {}".format(", ".join(self.dropout_parts)))

        if self.count_sum_feature:
            description_parts.append("using count sums")
        
        if self.early_stopping_rounds:
            description_parts.append("early stopping: " +
                "after {} epoch with no improvements".format(
                    self.early_stopping_rounds))
        
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
        encoder = dense_layers(
            inputs = self.x,
            num_outputs = self.hidden_sizes,
            activation_fn = relu,
            batch_normalisation = self.batch_normalisation, 
            is_training = self.is_training,
            input_dropout_keep_probability = self.dropout_keep_probability_x,
            hidden_dropout_keep_probability = self.dropout_keep_probability_h,
            scope = "ENCODER"
        )
        
        # Parameterising the approximate posterior and prior over z

        ## NOTE: tf.expand_dims(tf.expand_dims(x, 0), 0) allows broadcasting 
        ## on first, importance weight, and second, monte carlo, sample dim.
        
        for part_name in self.latent_distribution:
            part_distribution_name = self.latent_distribution[part_name]["name"]
            part_distribution = distributions[part_distribution_name]
            part_parameters = self.latent_distribution[part_name]["parameters"]
            
            with tf.variable_scope(part_name.upper()):
                
                # Retrieving layers for all latent distribution parameters
                for parameter_name in part_distribution["parameters"]:
                    if parameter_name in part_parameters:
                        part_parameters[parameter_name] = \
                            tf.expand_dims(tf.expand_dims(
                                tf.constant(
                                    part_parameters[parameter_name],
                                    dtype = tf.float32
                                ), 0), 0
                            )
                        continue
                    
                    parameter_activation_function = \
                        part_distribution["parameters"]\
                            [parameter_name]["activation function"]
                    p_min, p_max = \
                        part_distribution["parameters"]\
                            [parameter_name]["support"]
                    initial_value = \
                        part_distribution["parameters"]\
                            [parameter_name]["initial value"]
                    
                    def parameter_variable(num_outputs, name):
                        activation_fn = lambda x: tf.clip_by_value(
                            parameter_activation_function(x),
                            p_min + self.epsilon,
                            p_max - self.epsilon
                        )
                        if part_name == "posterior":
                            variable = dense_layer(
                                inputs = encoder,
                                num_outputs = num_outputs,
                                activation_fn = activation_fn,
                                is_training = self.is_training,
                                dropout_keep_probability = self.dropout_keep_probability_h,
                                scope = name
                            )
                        elif part_name == "prior":
                            variable = tf.expand_dims(tf.Variable(
                                activation_fn(initial_value([num_outputs])),
                                name = name
                            ), 0)
                        return variable
                    
                    # Switch: Use single or mixture (list) of distributions
                    if "mixture" in part_distribution_name:
                        if parameter_name == "logits":
                            logits = tf.expand_dims(tf.expand_dims(
                                parameter_variable(
                                    num_outputs =
                                        self.number_of_latent_clusters,
                                    name = parameter_name.upper()
                                ), 0), 0)
                            part_parameters[parameter_name] = logits
                        else:
                            part_parameters[parameter_name] = []
                            for k in range(self.number_of_latent_clusters):
                                part_parameters[parameter_name].append(
                                    tf.expand_dims(tf.expand_dims(
                                        parameter_variable(
                                            num_outputs = self.latent_size,
                                            name = parameter_name[:-1].upper()\
                                                + "_" + str(k)
                                        ), 0), 0)
                                )
                    else:
                        part_parameters[parameter_name] \
                            = tf.expand_dims(tf.expand_dims(
                                parameter_variable(
                                    num_outputs = self.latent_size,
                                    name = parameter_name.upper()
                                ), 0), 0)

        ## Latent posterior distribution
        ### Parameterise:
        if self.parameterise_latent_posterior:
            for parameter in self.latent_distribution["posterior"]["parameters"]:
                if isinstance(self.latent_distribution["posterior"]["parameters"]
                    [parameter], list):
                    for k in range(self.number_of_latent_clusters):
                        self.latent_distribution["posterior"]["parameters"]\
                            [parameter][k] += \
                            self.latent_distribution["prior"]["parameters"]\
                            [parameter][k]
                else:
                    self.latent_distribution["posterior"]["parameters"]\
                        [parameter] += \
                        self.latent_distribution["prior"]["parameters"]\
                        [parameter]
        
        self.q_z_given_x = \
            distributions[self.latent_distribution["posterior"]["name"]]\
                ["class"](self.latent_distribution["posterior"]["parameters"])
        
        ### Analytical mean:
        self.q_z_mean = self.q_z_given_x.mean()
        
        ### Sampling of
            ### 1st dim.: importance weighting samples
            ### 2nd dim.: monte carlo samples

        total_number_of_samples = tf.where(
            self.use_deterministic_z,
            1,
            self.number_of_iw_samples * self.number_of_mc_samples
        )

        self.z = tf.cast(
            tf.reshape(
                tf.cond(self.use_deterministic_z, 
                    lambda: tf.expand_dims(self.q_z_mean, 0),
                    lambda: self.q_z_given_x.sample(total_number_of_samples)
                ),
                [-1, self.latent_size]
            ), tf.float32
        )

        ## Latent prior distribution
        ### Parameterise:
        self.p_z = \
            distributions[self.latent_distribution["prior"]["name"]]\
                ["class"](self.latent_distribution["prior"]["parameters"])
        
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
                    tf.diag_part(tf.squeeze(self.p_z.components[k].covariance())))
        else:
            self.p_z_probabilities.append(tf.constant(1.))
            self.p_z_means.append(tf.squeeze(self.p_z.mean()))
            self.p_z_variances.append(tf.squeeze(self.p_z.stddev()))
        
        # Decoder - Generative model, p(x|z)
        
        # Make sure we use a replication pr. sample of the feature sum, 
        # when adding this to the features.  
        if self.count_sum:
            replicated_n = tf.tile(
                self.n,
                [self.number_of_iw_samples*self.number_of_mc_samples, 1]
                )

        if self.count_sum_feature:
            replicated_n_feature = tf.tile(
                self.n_feature,
                [self.number_of_iw_samples*self.number_of_mc_samples, 1]
            )
            decoder = tf.concat(
                [self.z, replicated_n_feature],
                axis = 1,
                name = 'Z_N'
            )
        else:
            decoder = self.z
        
        decoder = dense_layers(
            inputs = decoder,
            num_outputs = self.hidden_sizes,
            reverse_order = True,
            activation_fn = relu,
            batch_normalisation = self.batch_normalisation, 
            is_training = self.is_training,
            input_dropout_keep_probability = self.dropout_keep_probability_z,
            hidden_dropout_keep_probability = self.dropout_keep_probability_h,
            scope = "DECODER"
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
                    dropout_keep_probability = self.dropout_keep_probability_h,
                    scope = parameter.upper()
                )
            
            if "constrained" in self.reconstruction_distribution_name or \
                "multinomial" in self.reconstruction_distribution_name:
                self.p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta,
                    replicated_n
                )
            elif "multinomial" in self.reconstruction_distribution_name:
                self.p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta,
                    replicated_n
                )
            else:
                self.p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta
                )
            
            if self.k_max:
                x_logits = dense_layer(
                    inputs = decoder,
                    num_outputs = self.feature_size * self.number_of_reconstruction_classes,
                    activation_fn = None,
                    is_training = self.is_training,
                    dropout_keep_probability = self.dropout_keep_probability_h,
                    scope = "P_K"
                )
                
                x_logits = tf.reshape(x_logits,
                    [-1, self.feature_size, self.number_of_reconstruction_classes])
                
                self.p_x_given_z = Categorized(
                    dist = self.p_x_given_z,
                    cat = Categorical(logits = x_logits)
                )
            

            self.p_x_given_z_mean = tf.reshape(self.p_x_given_z.mean(), 
                [
                    self.number_of_iw_samples,
                    self.number_of_mc_samples,
                    -1,
                    self.feature_size
                ]
            )

            self.p_x_given_z_variance = tf.reshape(self.p_x_given_z.variance(), 
                [
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
    
    def loss(self):
        
        # # Recognition prior
        # if self.latent_distribution_name == "gaussian":
        #     p_z_mu = tf.constant(0.0, dtype = tf.float32)
        #     p_z_sigma = tf.constant(1.0, dtype = tf.float32)
        #     p_z = Normal(p_z_mu, p_z_sigma)
        # if self.latent_distribution_name == "gaussian mixture":
        #     p_z_mu = tf.constant(0.0, dtype = tf.float32)
        #     p_z_sigma = tf.constant(1.0, dtype = tf.float32)
        #     p_z = Normal(p_z_mu, p_z_sigma)
        # elif self.latent_distribution_name == "bernoulli":
        #     p_z_p = tf.constant(0.0, dtype = tf.float32)
        #     p_z = Bernoulli(p = p_z_p)
        
        # Prepare replicated and reshaped arrays
        ## Replicate out batches in tiles pr. sample into: 
        ### shape = (R * L * batchsize, D_x)
        t_tiled = tf.tile(self.t, [self.number_of_iw_samples*self.number_of_mc_samples, 1])
        ## Reshape samples back to: 
        ### shape = (R, L, batchsize, D_z)
        z_reshaped = tf.reshape(self.z, [self.number_of_iw_samples, self.number_of_mc_samples, -1, self.latent_size])

        # Loss
        ## Reconstruction error
        ### 1. Evaluate all log(p(x|z)) (R * L * batchsize, D_x) target values in the (R * L * batchsize, D_x) probability distributions learned
        ### 2. Sum over all N_x features
        ### 3. and reshape it back to (R, L, batchsize) 
        p_x_given_z_log_prob = self.p_x_given_z.log_prob(t_tiled)
        log_p_x_given_z = tf.reshape(
            tf.reduce_sum(
                p_x_given_z_log_prob,
                axis = -1
            ),
            [self.number_of_iw_samples, self.number_of_mc_samples, -1]
        )
        # (B, D_x)
        self.p_x_loglik = log_reduce_exp(
            tf.reshape(
                p_x_given_z_log_prob,
                [self.number_of_iw_samples*self.number_of_mc_samples,
                -1, self.feature_size]
            ),
            reduction_function=tf.reduce_mean,
            axis = 0
        )

        # self.p_x_count_loglik = tf.zeros(self.max_count)
        # loglik_condition = lambda i, loglik, count_loglik, t, max_count: i < 100
        # loglik_body = lambda i, loglik, count_loglik, t, max_count: (
        #     tf.add(i, 1),
        #     count_loglik[i].assign(
        #         tf.divide(
        #             tf.reduce_sum(
        #                 tf.where(
        #                     tf.equal(t, tf.cast(i, tf.float32)),
        #                     loglik,
        #                     tf.zeros_like(loglik)
        #                 )
        #             ),
        #             tf.cast(tf.reduce_sum(tf.cast(tf.equal(t, tf.cast(i, tf.float32)), tf.int32)), tf.float32)
        #         )
        #     )
        # )

        # i = tf.constant(0)

        # loglik_loop = tf.while_loop(
        #     loglik_condition,
        #     loglik_body,
        #     [i, self.p_x_loglik, self.p_x_count_loglik, self.t, self.max_count]
        # )

        # self.p_x_count_loglik_means = loglik_loop[2]

        # Average over all samples and examples and add to losses in summary
        self.ENRE = tf.reduce_mean(log_p_x_given_z)
        tf.add_to_collection('losses', self.ENRE)

        # Recognition error
        if "mixture" in self.latent_distribution_name:
            ## Evaluate Kullback-Leibler divergence numerically with sampling
            ### Evaluate all log(q(z|x)) and log(p(z)) on (R, L, batchsize, D_z) latent sample values.
            ### shape =  (R, L, batchsize, D_z)
            log_q_z_given_x = self.q_z_given_x.log_prob(z_reshaped)
            ### shape =  (R, L, batchsize, D_z)
            log_p_z = self.p_z.log_prob(z_reshaped)

            if "mixture" not in self.latent_distribution["posterior"]["name"]:
                log_q_z_given_x = tf.reduce_sum(log_q_z_given_x, axis = -1)
            if "mixture" not in self.latent_distribution["prior"]["name"]:
                log_p_z = tf.reduce_sum(log_p_z, axis = -1)
            
            # Kullback-Leibler Divergence:  KL[q(z|x)||p(z)] =
            # E_q[log(q(z|x)/p(z))] = E_q[log(q(z|x))] - E_q[log(p(z))]
            KL = log_q_z_given_x - log_p_z

            # Compute importance weighted estimate of 
            # E_x[p(x)] = E_{z_mc}[1/K_{iw} * sum(E_x[p(x|z_iw)]) p(z)/q(z|x) ] 
            # p(z)/q(z|x) = exp(-KL)
            # (batch_size, feature_size)
            # self.p_x_mean = tf.reduce_mean(self.p_x_given_z_mean*tf.exp(-KL), (0, 1))

            ## KL Regularisation term: 
            KL_qp = tf.reduce_mean(KL, name = "kl_divergence")
            tf.add_to_collection('losses', KL_qp)
            self.KL = KL_qp
            # Get mean KL for all D_z dim. --> shape = (1)
            self.KL_all = tf.expand_dims(tf.reduce_mean(KL), -1)
        else:
            if self.analytical_kl_term:
                ## Evaluate Kullback-Leibler divergence analytically without sampling
                KL =kl_divergence(self.q_z_given_x, self.p_z)
            else:
                ## Evaluate Kullback-Leibler divergence numerically with sampling
                ### Evaluate all log(q(z|x)) and log(p(z)) on (R, L, batchsize, D_z) latent sample values.
                ### shape =  (R, L, batchsize, D_z)
                log_q_z_given_x = self.q_z_given_x.log_prob(z_reshaped)
                ### shape =  (R, L, batchsize, D_z)
                log_p_z = self.p_z.log_prob(z_reshaped)

                # Kullback-Leibler Divergence:  KL[q(z|x)||p(z)] =
                # E_q[log(q(z|x)/p(z))] = E_q[log(q(z|x))] - E_q[log(p(z))]
                KL = log_q_z_given_x - log_p_z

            # Get mean KL for all D_z dim. --> shape = (D_z)
            self.KL_all = tf.reduce_mean(
                tf.reshape(KL, [-1, self.latent_size]),
                axis = 0
            )
            ## KL Regularisation term: Sum up KL over all latent dim.: shape --> () 
            KL_qp = tf.reduce_sum(self.KL_all, name = "kl_divergence")
            tf.add_to_collection('losses', KL_qp)
            self.KL = KL_qp

            # (R, L, B, D_x) --> (R, L, B)
            KL = tf.reduce_sum(KL, axis = -1)

            # From all z-samples compute importance weighted estimate of 
            # E_x[p(x)] = E_{z_mc}[1/R * sum(E_x[p(x_r|z_r)]) p(z_r)/q(z_r|x) ] 
            #   where: w_r = p(z_r)/q(z_r|x) = exp(-KL[q(z_r|x) || p(z_r)])
            # (R, L, B) --> (R, B, 1)
            # iw_weight_given_z = tf.expand_dims(
            #     tf.exp(
            #         tf.where(self.number_of_iw_samples > 1, 1.0, 0.0) * \
            #             - KL[:, 0, :]
            #     ),
            #     -1
            # ) 

            # Importance weighted Monte Carlo estimates of: 
            # Reconstruction mean (marginalised conditional mean): 
            ##      E[x] = E[E[x|z]] = E_q(z|x)[E_p(x|z)[x]]
            ##           = E_z[p_x_given_z.mean] = E_z[<x|z_{r,l}>]
            ##     \approx Ê[x] = 1/(R*L) \sum^R_r w_r \sum^L_{l=1} <x|z_{r,l}>
            ##          where <x|z_{r,l}> is explicitly known from p(x|z).

            # (R, L, B, D_x) --> (R, B, D_x) --> (B, D_x)
            self.p_x_mean = tf.reduce_mean(
                tf.reduce_mean(self.p_x_given_z_mean, 1), 
                0
            )

            # Estimated variance of likelihood expectation:
            # ^V[E[x|z]] = ( E[x|z_l] - Ê[x] )^2
            self.variance_of_p_x_given_z_mean = tf.reduce_mean(
                tf.reduce_mean(
                    tf.square(
                        self.p_x_given_z_mean - tf.reshape(
                            self.p_x_mean, 
                            [1, 1, -1, self.feature_size]
                        )
                    ),
                    1
                ), 
                0
            )

            self.stddev_of_p_x_given_z_mean = tf.sqrt(
                self.variance_of_p_x_given_z_mean
            )

            # Reconstruction standard deviation of Monte Carlo estimator: 
            ##      ^V[Ê[x]] = 1/(R*L) * V[E_p(x|z)[x]] = 1/(R*L)^2 * V[<x|z>]
            ##              = 1/(R*L) * Ê_z[(E[x|z] - E[x])^2]
            ##              = 1/(R*L) * Ê_z[(p_x_given_z.mean - Ê[x])^2]
            ##              = 1/(R*L) * ^V[E[x|z]]
            # (R, L, B, D_x)
            self.variance_of_p_x_mean = self.variance_of_p_x_given_z_mean / tf.cast(self.number_of_iw_samples * self.number_of_mc_samples, dtype=tf.float32)

            self.stddev_of_p_x_mean = tf.sqrt(self.variance_of_p_x_mean)

            ## Estimator for variance decomposition:
            # V[x] = E[V(x|z)] + V[E(x|z)] \approx ^V_z[E[x|z]] + Ê_z[V[x|z]]
            self.p_x_variance = self.variance_of_p_x_given_z_mean +\
                tf.reduce_mean(
                    tf.reduce_mean(
                        self.p_x_given_z_variance,
                        1
                    ), 
                    0
                )

            self.p_x_stddev = tf.sqrt(self.p_x_variance)

        # log-mean-exp (to avoid over- and underflow) over iw_samples dimension
        ## -> shape: (L, batch_size)
        LL = log_reduce_exp(log_p_x_given_z - self.warm_up_weight * KL, reduction_function=tf.reduce_mean, axis = 0)

        # average over eq_samples, batch_size dimensions    -> shape: ()
        self.lower_bound = tf.reduce_mean(LL) # scalar

        # # Averaging over samples.
        # self.lower_bound = tf.subtract(log_p_x_given_z, 
        #     tf.where(self.is_training, self.warm_up_weight * KL_qp, KL_qp), name = 'lower_bound')
        tf.add_to_collection('losses', self.lower_bound)
        self.ELBO = self.lower_bound
        
        # Add scalar summaries for the losses
        # for l in tf.get_collection('losses'):
        #     tf.summary.scalar(l.op.name, l)
    
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
            # self.train_op = optimiser.minimize(
            #     -self.lower_bound,
            #     global_step = self.global_step
            # )
        
            gradients = optimiser.compute_gradients(-self.lower_bound)
            clipped_gradients = [(tf.clip_by_value(gradient, -1., 1.), variable) for gradient, variable in gradients]
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
        reset_training = False, plot_for_every_n_epochs = False):
        
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
            n_train = training_set.count_sum
            n_valid = validation_set.count_sum

        if self.count_sum_feature:
            n_feature_train = training_set.normalised_count_sum
            n_feature_valid = validation_set.normalised_count_sum
        
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
        
        ## Learning curves
        
        learning_curves = {
            "training": {
                "lower_bound": [],
                "reconstruction_error": [],
                "kl_divergence": [],
            },
            "validation": {
                "lower_bound": [],
                "reconstruction_error": [],
                "kl_divergence": [],
            }
        }
        
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
                print("Restoring earlier model parameters.")
                restoring_time_start = time()
                
                self.saver.restore(session, checkpoint.model_checkpoint_path)
                epoch_start = int(os.path.split(
                    checkpoint.model_checkpoint_path)[-1].split('-')[-1])
                
                ELBO_valid_learning_curve = loadLearningCurves(self,
                    "validation")["lower_bound"]
                ELBO_valid_maximum = ELBO_valid_learning_curve.max()
                ELBO_valid_prev = ELBO_valid_learning_curve[-1]
                epochs_with_no_improvement = epochsWithNoImprovement(
                    ELBO_valid_learning_curve)
                ELBO_valid_early_stopping = ELBO_valid_learning_curve[
                    -1 - epochs_with_no_improvement]
                
                if os.path.exists(self.early_stopping_log_directory) \
                    and epochs_with_no_improvement == 0:
                    self.stopped_early = True
                else:
                    self.stopped_early = False
                
                restoring_duration = time() - restoring_time_start
                print("Earlier model parameters restored ({}).".format(
                    formatDuration(restoring_duration)))
                print()
            else:
                print("Initialising model parameters.")
                initialising_time_start = time()
                
                session.run(tf.global_variables_initializer())
                parameter_summary_writer.add_graph(session.graph)
                epoch_start = 0
                
                ELBO_valid_maximum = - numpy.inf
                ELBO_valid_prev = - numpy.inf
                epochs_with_no_improvement = 0
                ELBO_valid_early_stopping = - numpy.inf
                
                self.stopped_early = False
                
                initialising_duration = time() - initialising_time_start
                print("Model parameters initialised ({}).".format(
                    formatDuration(initialising_duration)))
                print()
            
            status["trained"] = "{}-{}".format(epoch_start, number_of_epochs)
            
            # Training loop
            
            data_string = dataString(training_set,
                self.reconstruction_distribution_name)
            print(trainingString(epoch_start, number_of_epochs, data_string))
            print()
            training_time_start = time()
            
            for epoch in range(epoch_start, number_of_epochs):
                
                if noisy_preprocess:
                    print("Noisily preprocess values.")
                    noisy_time_start = time()
                    x_train = noisy_preprocess(
                        training_set.preprocessed_values)
                    t_train = x_train
                    x_valid = noisy_preprocess(
                        validation_set.preprocessed_values)
                    t_valid = x_valid
                    noisy_duration = time() - noisy_time_start
                    print("Values noisily preprocessed ({}).".format(
                        formatDuration(noisy_duration)))
                    print()
                
                epoch_time_start = time()
                
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
                        self.use_deterministic_z: False,
                        self.learning_rate: learning_rate, 
                        self.warm_up_weight: warm_up_weight,
                        self.number_of_iw_samples: self.number_of_importance_samples["training"],
                        self.number_of_mc_samples: self.number_of_monte_carlo_samples["training"]
                    }
                    
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[batch_indices]

                    if self.count_sum_feature:
                        feed_dict_batch[self.n_feature] = n_feature_train[batch_indices]

                    # Run the stochastic batch training operation
                    _, batch_loss = session.run(
                        [self.train_op, self.lower_bound],
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
                                time() - training_time_start)
                            status["last epoch time"] = formatDuration(
                                time() - epoch_time_start)
                            return status
                
                print()
                
                epoch_duration = time() - epoch_time_start
                
                print("Epoch {} ({}):".format(epoch + 1,
                    formatDuration(epoch_duration)))

                # With warmup or not
                if warm_up_weight < 1:
                    print('    Warm-up weight: {:.2g}'.format(warm_up_weight))

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
                KL_train = 0
                ENRE_train = 0
                
                if "mixture" in self.latent_distribution_name: 
                    z_KL = numpy.zeros(1)                
                else:    
                    z_KL = numpy.zeros(self.latent_size)
                
                for i in range(0, M_train, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_train[subset]
                    t_batch = t_train[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.use_deterministic_z: False,
                        self.warm_up_weight: 1.0,
                        self.number_of_iw_samples: self.number_of_importance_samples["training"],
                        self.number_of_mc_samples: self.number_of_monte_carlo_samples["training"]
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[subset]

                    if self.count_sum_feature:
                        feed_dict_batch[self.n_feature] = n_feature_train[subset]

                    ELBO_i, KL_i, ENRE_i, z_KL_i = session.run(
                        [self.ELBO, self.KL, self.ENRE, self.KL_all],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_train += ELBO_i
                    KL_train += KL_i
                    ENRE_train += ENRE_i
                    
                    z_KL += z_KL_i
                
                ELBO_train /= M_train / batch_size
                KL_train /= M_train / batch_size
                ENRE_train /= M_train / batch_size
                
                z_KL /= M_train / batch_size
                
                learning_curves["training"]["lower_bound"].append(ELBO_train)
                learning_curves["training"]["reconstruction_error"].append(
                    ENRE_train)
                learning_curves["training"]["kl_divergence"].append(KL_train)
                
                evaluating_duration = time() - evaluating_time_start
                
                ## Summaries
                
                ### Losses
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_train)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_train)
                summary.value.add(tag="losses/kl_divergence",
                    simple_value = KL_train)
                
                ### KL divergence
                for i in range(z_KL.size):
                    summary.value.add(tag="kl_divergence_neurons/{}".format(i),
                        simple_value = z_KL[i])
                
                ### Writing
                training_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                training_summary_writer.flush()
                
                print("    Training set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                    ELBO_train, ENRE_train, KL_train))
                
                ## Validation
                
                evaluating_time_start = time()
                
                ELBO_valid = 0
                KL_valid = 0
                ENRE_valid = 0
                
                q_z_mean_valid = numpy.empty([M_valid, self.latent_size])
                
                for i in range(0, M_valid, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_valid[subset]
                    t_batch = t_valid[subset]
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
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_valid[subset]

                    if self.count_sum_feature:
                        feed_dict_batch[self.n_feature] = n_feature_valid[subset]

                    ELBO_i, KL_i, ENRE_i, q_z_mean_i = session.run(
                        [self.ELBO, self.KL, self.ENRE, self.q_z_mean],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_valid += ELBO_i
                    KL_valid += KL_i
                    ENRE_valid += ENRE_i
                    
                    q_z_mean_valid[subset] = q_z_mean_i
                
                ELBO_valid /= M_valid / batch_size
                KL_valid /= M_valid / batch_size
                ENRE_valid /= M_valid / batch_size
                
                learning_curves["validation"]["lower_bound"].append(ELBO_valid)
                learning_curves["validation"]["reconstruction_error"].append(
                    ENRE_valid)
                learning_curves["validation"]["kl_divergence"].append(KL_valid)
                
                ## Centroids
            
                p_z_probabilities, p_z_means, p_z_variances = \
                    session.run(
                    [self.p_z_probabilities, self.p_z_means,
                        self.p_z_variances]
                )
                
                ## Summaries
                
                ### Losses
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_valid)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_valid)
                summary.value.add(tag="losses/kl_divergence",
                    simple_value = KL_valid)
                
                ### Centroids
                for k in range(len(p_z_probabilities)):
                    summary.value.add(
                        tag="prior/cluster_{}/probability".format(k),
                        simple_value = p_z_probabilities[k]
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
                            tag="prior/cluster_{}/mean/dimension_{}".format(k, l),
                            simple_value = p_z_mean_k_l
                        )
                        summary.value.add(
                            tag="prior/cluster_{}/variance/dimension_{}"\
                                .format(k, l),
                            simple_value = p_z_variances_k_l
                        )
                
                ### Writing
                validation_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                validation_summary_writer.flush()
                
                evaluating_duration = time() - evaluating_time_start
                print("    Validation set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                    ELBO_valid, ENRE_valid, KL_valid))
                
                # Early stopping
                if not self.stopped_early:
                    
                    if ELBO_valid < ELBO_valid_early_stopping:
                        if epochs_with_no_improvement == 0:
                            print("    Early stopping:",
                                "Validation loss did not improve",
                                "for this epoch.")
                            print("        " + \
                                "Saving model parameters for previous epoch.")
                            saving_time_start = time()
                            ELBO_valid_early_stopping = ELBO_valid
                            current_checkpoint = \
                                tf.train.get_checkpoint_state(self.log_directory)
                            if current_checkpoint:
                                copyModelDirectory(current_checkpoint,
                                    self.early_stopping_log_directory)
                            saving_duration = time() - saving_time_start
                            print("        " + 
                                "Previous model parameters saved ({})."\
                                .format(formatDuration(saving_duration)))
                        else:
                            print("    Early stopping:",
                                "Validation loss has not improved",
                                "for {} epochs.".format(
                                    epochs_with_no_improvement + 1))
                        epochs_with_no_improvement += 1
                    else:
                        if epochs_with_no_improvement > 0:
                            print("    Early stopping cancelled:",
                                "Validation loss improved.")
                        epochs_with_no_improvement = 0
                        ELBO_valid_early_stopping = ELBO_valid
                        if os.path.exists(self.early_stopping_log_directory):
                            shutil.rmtree(self.early_stopping_log_directory)
                    
                    if epochs_with_no_improvement >= self.early_stopping_rounds:
                        print("    Early stopping in effect:",
                            "Previously saved model parameters is available.")
                        self.stopped_early = True
                        epochs_with_no_improvement = 0
                
                # Saving model parameters (update checkpoint)
                print('    Saving model parameters.')
                saving_time_start = time()
                self.saver.save(session, checkpoint_file,
                    global_step = epoch + 1)
                saving_duration = time() - saving_time_start
                print('    Model parameters saved ({}).'.format(
                    formatDuration(saving_duration)))
                
                # Saving best model parameters yet
                if ELBO_valid > ELBO_valid_maximum:
                    print("    Best validation ELBO yet.",
                        "Saving model parameters as best model parameters.")
                    saving_time_start = time()
                    ELBO_valid_maximum = ELBO_valid
                    current_checkpoint = \
                        tf.train.get_checkpoint_state(self.log_directory)
                    if current_checkpoint:
                        copyModelDirectory(current_checkpoint,
                            self.best_model_log_directory)
                    removeOldCheckpoints(self.best_model_log_directory)
                    saving_duration = time() - saving_time_start
                    print('    Best model parameters saved ({}).'.format(
                        formatDuration(saving_duration)))
                
                print()
                
                # Plot latent validation values
                if plot_for_every_n_epochs is None:
                    under_10 = epoch < 10
                    under_100 = epoch < 100 and (epoch + 1) % 10 == 0
                    under_1000 = epoch < 1000 and (epoch + 1) % 50 == 0 
                    above_1000 = epoch > 1000 and (epoch + 1) % 100 == 0 
                    last_one = epoch == number_of_epochs - 1
                    plot_intermediate_results = under_10 \
                        or under_100 \
                        or under_1000 \
                        or above_1000 \
                        or last_one
                else: 
                    plot_intermediate_results = \
                        epoch % plot_for_every_n_epochs == 0

                if plot_intermediate_results:
                    if "mixture" in self.latent_distribution_name:
                        K = len(p_z_probabilities)
                        L = self.latent_size
                        p_z_covariance_matrices = numpy.empty([K, L, L])
                        for k in range(K):
                            p_z_covariance_matrices[k] = numpy.diag(p_z_variances[k])
                        centroids = {
                            "prior": {
                                "probabilities": numpy.array(p_z_probabilities),
                                "means": numpy.stack(p_z_means),
                                "covariance_matrices": p_z_covariance_matrices
                            }
                        }
                    else:
                        centroids = None
                    analyseIntermediateResults(
                        learning_curves, epoch_start, epoch,
                        q_z_mean_valid, validation_set, centroids,
                        self.name, self.type,
                        self.main_results_directory
                    )
                    print()
                else:
                    analyseIntermediateResults(
                        learning_curves, epoch_start,
                        model_name = self.name,
                        model_type = self.type,
                        results_directory = self.main_results_directory
                    )
                    print()
                
                # Update variables for previous iteration
                ELBO_valid_prev = ELBO_valid
            
            training_duration = time() - training_time_start
            
            if epoch_start >= number_of_epochs:
                epoch_duration = training_duration
            else:
                print("Model trained for {} epochs ({}).".format(
                    number_of_epochs, formatDuration(training_duration)))
            
            # Clean up
            
            # removeOldCheckpoints(self.log_directory)
            
            status["completed"] = True
            status["training time"] = formatDuration(training_duration)
            status["last epoch time"] = formatDuration(epoch_duration)
            
            return status
    
    def evaluate(self, evaluation_set, batch_size = 100, predict_labels = False,
        use_early_stopping_model = False, use_best_model = False,
        use_deterministic_z = False):
        
        batch_size /= self.number_of_importance_samples["evaluation"] \
            * self.number_of_monte_carlo_samples["evaluation"]
        batch_size = int(numpy.ceil(batch_size))
        
        if self.count_sum:
            n_eval = evaluation_set.count_sum
        
        if self.count_sum_feature:
            n_feature_eval = evaluation_set.normalised_count_sum
        
        M_eval = evaluation_set.number_of_examples
        F_eval = evaluation_set.number_of_features
        
        noisy_preprocess = evaluation_set.noisy_preprocess
        
        if not noisy_preprocess:
            
            x_eval = evaluation_set.preprocessed_values
        
            if self.reconstruction_distribution_name == "bernoulli":
                t_eval = evaluation_set.binarised_values
            else:
                t_eval = evaluation_set.values
            
        else:
            print("Noisily preprocess values.")
            noisy_time_start = time()
            x_eval = noisy_preprocess(evaluation_set.preprocessed_values)
            t_eval = x_eval
            noisy_duration = time() - noisy_time_start
            print("Values noisily preprocessed ({}).".format(
                formatDuration(noisy_duration)))
            print()
    
        # max_count = int(max(t_eval, axis = (0, 1)))

        if use_early_stopping_model:
            log_directory = self.early_stopping_log_directory
        elif use_best_model:
            log_directory = self.best_model_log_directory
        else:
            log_directory = self.log_directory
            
        checkpoint = tf.train.get_checkpoint_state(log_directory)
        
        eval_summary_directory = os.path.join(log_directory, "evaluation")
        if os.path.exists(eval_summary_directory):
            shutil.rmtree(eval_summary_directory)
        
        with tf.Session(graph = self.graph) as session:
            
            eval_summary_writer = tf.summary.FileWriter(
                eval_summary_directory)
            
            if checkpoint:
                self.saver.restore(session, checkpoint.model_checkpoint_path)
                epoch = int(os.path.split(
                    checkpoint.model_checkpoint_path)[-1].split('-')[-1])
            else:
                print("Cannot evaluate model when it has not been trained.")
                return None, None, None
            
            data_string = dataString(evaluation_set,
                self.reconstruction_distribution_name)
            print('Evaluating trained model on {}.'.format(data_string))
            evaluating_time_start = time()
            
            ELBO_eval = 0
            KL_eval = 0
            ENRE_eval = 0
            
            p_x_mean_eval = numpy.empty([M_eval, F_eval])
            p_x_stddev_eval = numpy.empty([M_eval, F_eval])
            stddev_of_p_x_mean_eval = numpy.empty([M_eval, F_eval])
            p_x_loglik = numpy.empty([M_eval, F_eval])
            # p_x_count_loglik_means = numpy.zeros(max_count)

            q_z_mean_eval = numpy.empty([M_eval, self.latent_size])
            
            if use_deterministic_z:
                number_of_iw_samples = 1
                number_of_mc_samples = 1
            else:
                number_of_iw_samples = self.number_of_importance_samples["evaluation"]
                number_of_mc_samples = self.number_of_monte_carlo_samples["evaluation"]

            for i in range(0, M_eval, batch_size):
                subset = slice(i, (i + batch_size))
                x_batch = x_eval[subset]
                t_batch = t_eval[subset]
                feed_dict_batch = {
                    self.x: x_batch,
                    self.t: t_batch,
                    self.is_training: False,
                    self.use_deterministic_z: use_deterministic_z,
                    self.warm_up_weight: 1.0,
                    self.number_of_iw_samples: number_of_iw_samples,
                    self.number_of_mc_samples: number_of_mc_samples
                }
                if self.count_sum:
                    feed_dict_batch[self.n] = n_eval[subset]
                
                if self.count_sum_feature:
                    feed_dict_batch[self.n_feature] = n_feature_eval[subset]
                
                ELBO_i, KL_i, ENRE_i, p_x_mean_i, p_x_stddev_i, stddev_of_p_x_mean_i, p_x_loglik_i, q_z_mean_i = session.run(
                    [self.ELBO, self.KL, self.ENRE,
                        self.p_x_mean, self.p_x_stddev,
                        self.stddev_of_p_x_given_z_mean, self.p_x_loglik,
                        self.q_z_mean],
                    feed_dict = feed_dict_batch
                )
                
                ELBO_eval += ELBO_i
                KL_eval += KL_i
                ENRE_eval += ENRE_i
                
                # Save Importance weighted Monte Carlo estimates of: 
                # Reconstruction mean (marginalised conditional mean): 
                ##      E[x] = E[E[x|z]] = E_q(z|x)[E_p(x|z)[x]]
                ##           = E_z[p_x_given_z.mean]
                ##     \approx 1/(R*L) \sum^R_r w_r \sum^L_{l=1} p_x_given_z.mean 
                p_x_mean_eval[subset] = p_x_mean_i

                # Reconstruction standard deviation: 
                ##      sqrt(V[x]) = sqrt(E[V[x|z]] + V[E[x|z]])
                ##      = E_z[p_x_given_z.var] + E_z[(p_x_given_z.mean - E[x])^2]
                p_x_stddev_eval[subset] = p_x_stddev_i

                # Estimated standard deviation of Monte Carlo estimate E[x].
                stddev_of_p_x_mean_eval[subset] = stddev_of_p_x_mean_i

                p_x_loglik[subset] = p_x_loglik_i
                # p_x_count_loglik_means += p_x_count_loglik_means_i

                q_z_mean_eval[subset] = q_z_mean_i
            
            ELBO_eval /= M_eval / batch_size
            KL_eval /= M_eval / batch_size
            ENRE_eval /= M_eval / batch_size
            
            ## Centroids
            
            p_z_probabilities, p_z_means, p_z_variances = session.run(
                [self.p_z_probabilities, self.p_z_means, self.p_z_variances]
            )
            

            ## Summaries
            
            summary = tf.Summary()
            summary.value.add(tag="losses/lower_bound",
                simple_value = ELBO_eval)
            summary.value.add(tag="losses/reconstruction_error",
                simple_value = ENRE_eval)
            summary.value.add(tag="losses/kl_divergence",
                simple_value = KL_eval)
            
            for k in range(len(p_z_probabilities)):
                summary.value.add(
                    tag="prior/cluster_{}/probability".format(k),
                    simple_value = p_z_probabilities[k]
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
                        tag="prior/cluster_{}/mean/dimension_{}".format(k, l),
                        simple_value = p_z_mean_k_l
                    )
                    summary.value.add(
                        tag="prior/cluster_{}/variance/dimension_{}"\
                            .format(k, l),
                        simple_value = p_z_variances_k_l
                    )
            
            eval_summary_writer.add_summary(summary, global_step = epoch)
            eval_summary_writer.flush()
            
            evaluating_duration = time() - evaluating_time_start
            print("    {} set ({}): ".format(
                evaluation_set.kind.capitalize(),
                formatDuration(evaluating_duration)) + \
                "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                ELBO_eval, ENRE_eval, KL_eval))
            
            # Data sets
            
            if noisy_preprocess or \
                self.reconstruction_distribution_name == "bernoulli":
                
                transformed_evaluation_set = DataSet(
                    name = evaluation_set.name,
                    values = t_eval,
                    preprocessed_values = None,
                    labels = evaluation_set.labels,
                    example_names = evaluation_set.example_names,
                    feature_names = evaluation_set.feature_names,
                    feature_selection = evaluation_set.feature_selection,
                    example_filter = evaluation_set.example_filter,
                    preprocessing_methods = evaluation_set.preprocessing_methods,
                    kind = evaluation_set.kind,
                    version = "transformed"
                )
            else:
                transformed_evaluation_set = evaluation_set
            
            reconstructed_evaluation_set = DataSet(
                name = evaluation_set.name,
                values = p_x_mean_eval,
                total_standard_deviations = p_x_stddev_eval,
                explained_standard_deviations = stddev_of_p_x_mean_eval,
                preprocessed_values = None,
                labels = evaluation_set.labels,
                example_names = evaluation_set.example_names,
                feature_names = evaluation_set.feature_names,
                feature_selection = evaluation_set.feature_selection,
                example_filter = evaluation_set.example_filter,
                preprocessing_methods = evaluation_set.preprocessing_methods,
                kind = evaluation_set.kind,
                version = "reconstructed"
            )

            likelihood_evaluation_set = DataSet(
                name = evaluation_set.name,
                values = numpy.exp(p_x_loglik),
                preprocessed_values = None,
                labels = evaluation_set.labels,
                example_names = evaluation_set.example_names,
                feature_names = evaluation_set.feature_names,
                feature_selection = evaluation_set.feature_selection,
                example_filter = evaluation_set.example_filter,
                preprocessing_methods = evaluation_set.preprocessing_methods,
                kind = evaluation_set.kind,
                version = "likelihood"
            )
            
            z_evaluation_set = DataSet(
                name = evaluation_set.name,
                values = q_z_mean_eval,
                preprocessed_values = None,
                labels = evaluation_set.labels,
                example_names = evaluation_set.example_names,
                feature_names = numpy.array(["latent variable {}".format(
                    i + 1) for i in range(self.latent_size)]),
                feature_selection = evaluation_set.feature_selection,
                example_filter = evaluation_set.example_filter,
                preprocessing_methods = evaluation_set.preprocessing_methods,
                kind = evaluation_set.kind,
                version = "z"
            )
            
            latent_evaluation_sets = {
                "z": z_evaluation_set
            }
            
            return transformed_evaluation_set, reconstructed_evaluation_set,\
                likelihood_evaluation_set, latent_evaluation_sets
