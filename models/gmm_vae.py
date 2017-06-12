import tensorflow as tf

from models.auxiliary import dense_layer, dense_layers, log_reduce_exp, reduce_logmeanexp

from tensorflow.python.ops.nn import relu, softmax
from tensorflow import sigmoid, identity

from tensorflow.contrib.distributions import Normal, Bernoulli, kl, Categorical
from distributions import distributions, latent_distributions, Categorized

import numpy
from numpy import inf

import copy
import os, shutil
from time import time
from auxiliary import formatDuration, normaliseString

from data import DataSet
from analysis import analyseIntermediateResults

class GaussianMixtureVariationalAutoEncoder_alternative(object):
    def __init__(self, feature_size, latent_size, hidden_sizes,
        number_of_monte_carlo_samples,
        number_of_importance_samples,
        analytical_kl_term = False,
        latent_distribution = "gaussian mixture",
        number_of_latent_clusters = 1,
        reconstruction_distribution = None,
        number_of_reconstruction_classes = None,
        batch_normalisation = True, count_sum = True,
        number_of_warm_up_epochs = 0, epsilon = 1e-6,
        log_directory = "log",
        results_directory = "results"):
        
        # Class setup
        super(GaussianMixtureVariationalAutoEncoder_alternative, self).__init__()
        
        self.type = "GMVAE_alt"
        
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        self.latent_distribution_name = latent_distribution
        self.latent_distribution = copy.deepcopy(
            latent_distributions[latent_distribution]
        )
        self.K = number_of_latent_clusters
        self.number_of_latent_clusters = number_of_latent_clusters
        self.analytical_kl_term = analytical_kl_term
        
        # Dictionary holding number of samples needed for the "monte carlo" 
        # estimator and "importance weighting" during both "train" and "test" time.  
        self.number_of_importance_samples = number_of_importance_samples
        self.number_of_monte_carlo_samples = number_of_monte_carlo_samples

        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = distributions\
            [reconstruction_distribution]
        
        self.k_max = number_of_reconstruction_classes
        
        self.batch_normalisation = batch_normalisation

        self.count_sum_feature = count_sum
        self.count_sum = self.count_sum_feature or "constrained" in \
            self.reconstruction_distribution_name or "multinomial" in \
            self.reconstruction_distribution_name

        self.number_of_warm_up_epochs = number_of_warm_up_epochs

        self.epsilon = epsilon
        
        self.main_log_directory = log_directory
        self.main_results_directory = results_directory
        
        # Graph setup
        
        self.graph = tf.Graph()
        
        self.parameter_summary_list = []
        
        with self.graph.as_default():
            
            self.x = tf.placeholder(tf.float32, [None, self.feature_size], 'X')
            self.t = tf.placeholder(tf.float32, [None, self.feature_size], 'T')
            
            
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            
            self.warm_up_weight = tf.placeholder(tf.float32, [], 'warm_up_weight')
            parameter_summary = tf.summary.scalar('warm_up_weight',
                self.warm_up_weight)
            self.parameter_summary_list.append(parameter_summary)
            
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')
            
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
            # Sum up counts in replicated_n feature if needed
            if self.count_sum or self.count_sum_feature:
                self.n = tf.placeholder(tf.float32, [None, 1], 'N')
                self.replicated_n = tf.tile(
                    self.n,
                    [self.number_of_iw_samples*self.number_of_mc_samples, 1]
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
            latent_part += "_c_" + str(self.K)
        
        reconstruction_part = normaliseString(
            self.reconstruction_distribution_name)
        
        if self.k_max:
            reconstruction_part += "_c_" + str(self.k_max)
        
        if self.count_sum_feature:
            reconstruction_part += "_sum"
        
        reconstruction_part += "_l_" + str(self.latent_size) \
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
                self.K))
        
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
        
        if self.analytical_kl_term:
            description_parts.append("using analytical KL term")
        
        if self.batch_normalisation:
            description_parts.append("using batch normalisation")
        if self.count_sum_feature:
            description_parts.append("using count sums")
        
        description = "\n    ".join(description_parts)
        
        return description
    
    @property
    def parameters(self, trainable = True):
        
        if trainable:
            
            parameters_string_parts = ["Trainable parameters"]
            
            with self.graph.as_default():
                trainable_parameters = tf.trainable_variables()
            
            width = max(map(len, [p.name for p in trainable_parameters]))
            
            for parameter in trainable_parameters:
                parameters_string_parts.append("{:{}}  {}".format(
                    parameter.name, width, parameter.get_shape()))
            
            parameters_string = "\n    ".join(parameters_string_parts)
        
        else:
            raise NotImplementedError("Can only return trainable parameters.")
        
        return parameters_string
    
    def q_z_given_x_y_graph(self, x, y, distribution_name = "modified gaussian", reuse = False):
        ## Encoder for q(z|x,y_i=1) = N(mu(x,y_i=1), sigma^2(x,y_i=1))
        with tf.variable_scope("Q"):
            distribution = distributions[distribution_name]
            xy = tf.concat((self.x, y), axis=-1)
            encoder = dense_layers(
                inputs = xy,
                num_outputs = self.hidden_sizes,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation, 
                is_training = self.is_training,
                scope = "ENCODER",
                reuse = reuse
            )

            with tf.variable_scope(normaliseString(distribution_name).upper()):
                # Loop over and add parameter layers to theta dict.
                theta = {}
                for parameter in distribution["parameters"]:
                    parameter_activation_function = \
                        distribution["parameters"]\
                        [parameter]["activation function"]
                    p_min, p_max = distribution["parameters"][parameter]["support"]
                    theta[parameter] = tf.expand_dims(tf.expand_dims(
                        dense_layer(
                            inputs = encoder,
                            num_outputs = self.latent_size,
                            activation_fn = lambda x: tf.clip_by_value(
                                parameter_activation_function(x),
                                p_min + self.epsilon,
                                p_max - self.epsilon
                            ),
                            is_training = self.is_training,
                            scope = parameter.upper(),
                            reuse = reuse
                    ), 0), 0)

                ### Parameterise:
                q_z_given_x_y = distribution["class"](theta)

                ### Analytical mean:
                z_mean = q_z_given_x_y.mean()

                ### Sampling of
                    ### 1st dim.: importance weighting samples
                    ### 2nd dim.: monte carlo samples
                z_samples = q_z_given_x_y.sample(
                            self.number_of_iw_samples * self.number_of_mc_samples
                            )
                z = tf.cast(
                    tf.reshape(
                        z_samples,
                        [-1, self.latent_size]
                    ), tf.float32)
        return q_z_given_x_y, z_mean, z

    def p_z_given_y_graph(self, y, distribution_name = "modified gaussian", reuse = False):
        with tf.variable_scope("P"):
            with tf.variable_scope(normaliseString(distribution_name).upper()):
                distribution = distributions[distribution_name]
                # Loop over and add parameter layers to theta dict.
                theta = {}
                for parameter in distribution["parameters"]:
                    parameter_activation_function = \
                        distribution["parameters"]\
                        [parameter]["activation function"]
                    p_min, p_max = distribution["parameters"][parameter]["support"]
                    theta[parameter] = tf.expand_dims(tf.expand_dims(
                        dense_layer(
                            inputs = y,
                            num_outputs = self.latent_size,
                            activation_fn = lambda x: tf.clip_by_value(
                                parameter_activation_function(x),
                                p_min + self.epsilon,
                                p_max - self.epsilon
                            ),
                            is_training = self.is_training,
                            scope = parameter.upper(),
                            reuse = reuse
                    ), 0), 0)

                p_z_given_y = distribution["class"](theta)
                p_z_mean = tf.reduce_mean(p_z_given_y.mean())
        return p_z_given_y, p_z_mean

    def q_y_given_x_graph(self, x, distribution_name = "categorical", reuse = False):
        with tf.variable_scope(distribution_name.upper()):
            distribution = distributions[distribution_name]
            ## Encoder
            encoder = dense_layers(
                inputs = self.x,
                num_outputs = self.hidden_sizes,
                activation_fn = relu,
                batch_normalisation = self.batch_normalisation, 
                is_training = self.is_training,
                scope = "ENCODER",
                reuse = reuse
            )
            # Loop over and add parameter layers to theta dict.
            theta = {}
            for parameter in distribution["parameters"]:
                parameter_activation_function = \
                    distribution["parameters"]\
                    [parameter]["activation function"]
                p_min, p_max = distribution["parameters"][parameter]["support"]
                theta[parameter] = dense_layer(
                        inputs = encoder,
                        num_outputs = self.K,
                        activation_fn = lambda x: tf.clip_by_value(
                            parameter_activation_function(x),
                            p_min + self.epsilon,
                            p_max - self.epsilon
                        ),
                        is_training = self.is_training,
                        scope = parameter.upper(),
                        reuse = reuse
                )
            ### Parameterise q(y|x) = Cat(pi(x))
            q_y_given_x = distribution["class"](theta)
        return q_y_given_x 

    def p_x_given_z_graph(self, z, reuse = False):
        # Decoder - Generative model, p(x|z)
        
        # Make sure we use a replication pr. sample of the feature sum, 
        # when adding this to the features.  
        if self.count_sum_feature:
            decoder = tf.concat([z, self.replicated_n], axis = -1, name = 'Z_N')
        else:
            decoder = z
        
        decoder = dense_layers(
            inputs = decoder,
            num_outputs = self.hidden_sizes[::-1],
            activation_fn = relu,
            batch_normalisation = self.batch_normalisation,
            is_training = self.is_training,
            scope = "DECODER",
            reuse = reuse
        )

        # Reconstruction distribution parameterisation
        
        with tf.variable_scope("DISTRIBUTION"):
            
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
                    scope = parameter.upper(),
                    reuse = reuse
                )
            
            if "constrained" in self.reconstruction_distribution_name or \
                "multinomial" in self.reconstruction_distribution_name:
                p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta,
                    self.replicated_n
                )
            elif "multinomial" in self.reconstruction_distribution_name:
                p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta,
                    self.replicated_n
                )
            else:
                p_x_given_z = self.reconstruction_distribution["class"](
                    x_theta
                )
            
            if self.k_max:
                x_logits = dense_layer(
                    inputs = decoder,
                    num_outputs = self.feature_size * self.k_max,
                    activation_fn = None,
                    is_training = self.is_training,
                    scope = "P_K",
                    reuse = reuse
                )
                
                x_logits = tf.reshape(x_logits,
                    [-1, self.feature_size, self.k_max])
                
                p_x_given_z = Categorized(
                    dist = p_x_given_z,
                    cat = Categorical(logits = x_logits)
                )
            
            return p_x_given_z

    def inference(self):
        # Retrieving layers parameterising all distributions in model:
        
        # Y latent space
        with tf.variable_scope("Y"):
            ## q(y|x) = Cat(pi(x))
            self.y_ = tf.fill(tf.stack(
                [tf.shape(self.x)[0],
                self.K]
                ), 0.0)
            y = [tf.add(self.y_, tf.constant(numpy.eye(
                self.K)[k], name='hot_at_{:d}'.format(k), dtype = tf.float32
            )) for k in range(self.K)]
            
            self.q_y_given_x = self.q_y_given_x_graph(self.x)
            self.q_z_probabilities = tf.reduce_mean(self.q_y_given_x.probs, 0) 
        
        # Z latent space
        with tf.variable_scope("Z"):
            self.q_z_given_x_y = [None]*self.K
            z_mean = [None]*self.K
            self.z = [None]*self.K
            self.p_z_given_y = [None]*self.K
            self.p_z_mean = [None]*self.K
            self.p_z_probabilities = tf.ones(self.K)/self.K
            self.p_z_means = []
            self.p_z_variances = []
            self.q_z_means = []
            self.q_z_variances = []
            # Loop over parameter layers for all K gaussians.
            for k in range(self.K):
                if k >= 1:
                    reuse_weights = True
                else:
                    reuse_weights = False

                ## Latent prior distribution
                self.q_z_given_x_y[k], z_mean[k], self.z[k] = \
                self.q_z_given_x_y_graph(self.x, y[k], reuse = reuse_weights) 
                # Latent prior distribution
                self.p_z_given_y[k], self.p_z_mean[k] = self.p_z_given_y_graph(y[k], reuse = reuse_weights)

                self.p_z_means.append(
                    tf.reduce_mean(self.p_z_given_y[k].mean(), [0, 1, 2]))
                self.p_z_variances.append(
                    tf.square(tf.reduce_mean(self.p_z_given_y[k].stddev(), [0, 1, 2])))

                self.q_z_means.append(
                    tf.reduce_mean(self.q_z_given_x_y[k].mean(), [0, 1, 2]))
                self.q_z_variances.append(
                    tf.reduce_mean(tf.square(self.q_z_given_x_y[k].stddev()), [0, 1, 2]))                    
            self.z_mean = tf.add_n([z_mean[k] * tf.expand_dims(tf.one_hot(tf.argmax(self.q_y_given_x.probs, -1), self.K)[:, k], -1) for k in range(self.K)])
        # Decoder for X 
        with tf.variable_scope("X"):
            self.p_x_given_z = [None]*self.K
            self.x_given_y_mean = [None]*self.K 
            for k in range(self.K):
                if k >= 1:
                    reuse_weights = True
                else:
                    reuse_weights = False

                self.p_x_given_z[k] = self.p_x_given_z_graph(self.z[k], reuse = reuse_weights)
                self.x_given_y_mean[k] = tf.reduce_mean(
                    tf.reshape(
                        self.p_x_given_z[k].mean(),
                        [self.number_of_iw_samples*\
                        self.number_of_mc_samples, -1, self.feature_size]
                    ),
                    axis = 0
                ) * tf.expand_dims(self.q_y_given_x.probs[:, k], -1)
            self.x_mean = tf.add_n(self.x_given_y_mean)
        
        # (B, K)
        self.y_mean = self.q_y_given_x.probs
        
        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            parameter_summary = tf.summary.histogram(parameter.name, parameter)
            self.parameter_summary_list.append(parameter_summary)
        self.parameter_summary = tf.summary.merge(self.parameter_summary_list)
    

    def loss(self):
        # Prepare replicated and reshaped arrays
        ## Replicate out batches in tiles pr. sample into: 
        ### shape = (N_iw * N_mc * batchsize, N_x)
        t_tiled = tf.tile(self.t, [self.number_of_iw_samples*self.number_of_mc_samples, 1])
        ## Reshape samples back to: 
        ### shape = (N_iw, N_mc, batchsize, N_z)
        z_reshaped = [tf.reshape(self.z[k], [self.number_of_iw_samples, self.number_of_mc_samples, -1, self.latent_size]) for k in range(self.K)]

        # H[q(y|x)] = -E_{q(y|x)}[ log(q(y|x)) ]
        # (B)
        q_y_given_x_entropy = self.q_y_given_x.entropy()
        # H[q(y|x)||p(y)] = -E_{q(y|x)}[ log(p(y)) ] = -E_{q(y|x)}[ log(1/K) ] = log(K)
        # ()
        p_y_cross_entropy = numpy.log(self.K)
        # KL(q||p) = -E_q(y|x)[log p(y)/q(y|x)] = -E_q(y|x)[log p(y)] + E_q(y|x)[log q(y|x)] = H(q|p) - H(q)
        # (B)
        KL_y = p_y_cross_entropy - q_y_given_x_entropy

        KL_z = [None] * self.K
        KL_z_mean = [None] * self.K
        log_p_x_given_z_mean = [None] * self.K
        log_likelihood_x_z = [None] * self.K

        for k in range(self.K):
            log_q_z_given_x_y = tf.reduce_sum(
                self.q_z_given_x_y[k].log_prob(
                    z_reshaped[k]
                ),
                axis = -1
            )
            log_p_z_given_y = tf.reduce_sum(
                self.p_z_given_y[k].log_prob(
                    z_reshaped[k]
                ),
                axis = -1
            )
            KL_z[k] = log_q_z_given_x_y - log_p_z_given_y
            KL_z_mean[k] = tf.reduce_mean(
                KL_z[k], 
                axis=(0,1)
            ) * self.q_y_given_x.probs[:, k] 

            log_p_x_given_z = tf.reshape(
                tf.reduce_sum(
                    self.p_x_given_z[k].log_prob(t_tiled), 
                    axis=-1
                ),
                [self.number_of_iw_samples, self.number_of_mc_samples, -1]
            )
            log_p_x_given_z_mean[k] = tf.reduce_mean(
                log_p_x_given_z,
                axis = (0,1)
            ) * self.q_y_given_x.probs[:, k] 
            # Monte carlo estimates over: 
                # Importance weight estimates using log-mean-exp 
                    # (to avoid over- and underflow) 
                    # shape: (N_mc, batch_size)
            ##  -> shape: (batch_size)
            log_likelihood_x_z[k] = tf.reduce_mean(
                log_reduce_exp(
                    log_p_x_given_z - self.warm_up_weight * KL_z[k],
                    reduction_function=tf.reduce_mean,
                    axis = 0
                ),
                axis = 0
            ) * self.q_y_given_x.probs[:, k]


        log_likelihood_x_z_sum = tf.add_n(log_likelihood_x_z)

        # self.lower_bound = tf.reduce_mean(
        #     log_likelihood_x_z_sum - KL_y
        # )
        # (B) --> ()
        self.KL_z = tf.reduce_mean(tf.add_n(KL_z_mean))
        self.KL_y = tf.reduce_mean(KL_y)
        self.KL = self.KL_z + self.KL_y
        self.KL_all = tf.expand_dims(self.KL, -1)
        self.ENRE = tf.reduce_mean(tf.add_n(log_p_x_given_z_mean))
        self.lower_bound = self.ENRE - self.KL
        self.ELBO = self.lower_bound
        tf.add_to_collection('losses', self.lower_bound)
        
    
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
        reset_training = False):
        
        # Logging
        
        status = {
            "completed": False,
            "message": None
        }
        
        # parameter_values = "lr_{:.1g}".format(learning_rate)
        # parameter_values += "_b_" + str(batch_size)
        
        # self.log_directory = os.path.join(self.log_directory, parameter_values)
        
        if reset_training and os.path.exists(self.log_directory):
            shutil.rmtree(self.log_directory)
        
        checkpoint_file = os.path.join(self.log_directory, 'model.ckpt')
        
        # Setup
        
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
            
            # Training loop
            
            if epoch_start == number_of_epochs:
                print("Model has already been trained for {} epochs.".format(
                    number_of_epochs))
            
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
                        self.number_of_iw_samples: self.number_of_importance_samples["training"],
                        self.number_of_mc_samples: self.number_of_monte_carlo_samples["training"]
                    }
                    
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[batch_indices]
                    
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
                KL_z_train = 0
                KL_y_train = 0
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
                        self.warm_up_weight: 1.0,
                        self.number_of_iw_samples: self.number_of_importance_samples["evaluation"],
                        self.number_of_mc_samples: self.number_of_monte_carlo_samples["evaluation"]
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[subset]
                    
                    ELBO_i, ENRE_i, KL_z_i, KL_y_i, z_KL_i = session.run(
                        [self.ELBO, self.ENRE,  self.KL_z, self.KL_y, self.KL_all],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_train += ELBO_i
                    KL_z_train += KL_z_i
                    KL_y_train += KL_y_i
                    ENRE_train += ENRE_i
                    
                    z_KL += z_KL_i
                
                ELBO_train /= M_train / batch_size
                KL_z_train /= M_train / batch_size
                KL_y_train /= M_train / batch_size
                ENRE_train /= M_train / batch_size
                
                z_KL /= M_train / batch_size
                
                evaluating_duration = time() - evaluating_time_start
                
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_train)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_train)
                summary.value.add(tag="losses/kl_divergence_z",
                    simple_value = KL_z_train)
                summary.value.add(tag="losses/kl_divergence_y",
                    simple_value = KL_y_train)

                for i in range(z_KL.size):
                    summary.value.add(tag="kl_divergence_neurons/{}".format(i),
                        simple_value = z_KL[i])
                
                training_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                training_summary_writer.flush()
                
                print("    Training set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL_z: {:.5g}, KL_y: {:.5g}.".format(
                    ELBO_train, ENRE_train, KL_z_train, KL_y_train))
                
                ## Validation
                
                evaluating_time_start = time()
                
                ELBO_valid = 0
                KL_z_valid = 0
                KL_y_valid = 0
                ENRE_valid = 0
                q_z_probabilities = numpy.zeros(self.K)
                q_z_means = numpy.zeros((self.K, self.latent_size))
                q_z_variances = numpy.zeros((self.K, self.latent_size))
                p_z_probabilities = numpy.zeros(self.K)
                p_z_means = numpy.zeros((self.K, self.latent_size))
                p_z_variances = numpy.zeros((self.K, self.latent_size))
                z_mean_valid = numpy.zeros((M_valid, self.latent_size))

                for i in range(0, M_valid, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_valid[subset]
                    t_batch = t_valid[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.warm_up_weight: 1.0,
                        self.number_of_iw_samples:
                            self.number_of_importance_samples["evaluation"],
                        self.number_of_mc_samples:
                            self.number_of_monte_carlo_samples["evaluation"]
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_valid[subset]
                    
                    ELBO_i, ENRE_i, KL_z_i, KL_y_i, \
                    q_z_probabilities_i, q_z_means_i, q_z_variances_i, p_z_probabilities_i, p_z_means_i, p_z_variances_i, z_mean_i = session.run(
                        [self.ELBO, self.ENRE, self.KL_z, self.KL_y, self.q_z_probabilities, self.q_z_means, self.q_z_variances, self.p_z_probabilities, self.p_z_means, self.p_z_variances, self.z_mean],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_valid += ELBO_i
                    KL_z_valid += KL_z_i
                    KL_y_valid += KL_y_i
                    ENRE_valid += ENRE_i
                    q_z_probabilities += numpy.array(q_z_probabilities_i)
                    q_z_means += numpy.array(q_z_means_i)
                    q_z_variances += numpy.array(q_z_variances_i)
                    p_z_probabilities += numpy.array(p_z_probabilities_i)
                    p_z_means += numpy.array(p_z_means_i)
                    p_z_variances += numpy.array(p_z_variances_i)
                    z_mean_valid[subset] = z_mean_i 
                
                ELBO_valid /= M_valid / batch_size
                KL_z_valid /= M_valid / batch_size
                KL_y_valid /= M_valid / batch_size
                ENRE_valid /= M_valid / batch_size
                q_z_probabilities /= M_valid / batch_size
                q_z_means /= M_valid / batch_size
                q_z_variances /= M_valid / batch_size
                p_z_probabilities /= M_valid / batch_size
                p_z_means /= M_valid / batch_size
                p_z_variances /= M_valid / batch_size
                
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_valid)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_valid)
                summary.value.add(tag="losses/kl_divergence_z",
                    simple_value = KL_z_valid)
                summary.value.add(tag="losses/kl_divergence_y",
                    simple_value = KL_y_valid)

                for k in range(self.K):
                    summary.value.add(
                        tag="prior/cluster_{}/probability".format(k),
                        simple_value = p_z_probabilities[k]
                    )
                    summary.value.add(
                        tag="posterior/cluster_{}/probability".format(k),
                        simple_value = q_z_probabilities[k]
                    )
                    for l in range(self.latent_size):
                        summary.value.add(
                            tag="prior/cluster_{}/mean/dimension_{}".format(k, l),
                            simple_value = p_z_means[k][l]
                        )
                        summary.value.add(
                            tag="posterior/cluster_{}/mean/dimension_{}".format(k, l),
                            simple_value = q_z_means[k, l]
                        )
                        summary.value.add(
                            tag="prior/cluster_{}/variance/dimension_{}"\
                                .format(k, l),
                            simple_value = p_z_variances[k][l]
                        )
                        summary.value.add(
                            tag="posterior/cluster_{}/variance/dimension_{}"\
                                .format(k, l),
                            simple_value = q_z_variances[k, l]
                        )

                validation_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                validation_summary_writer.flush()
                
                evaluating_duration = time() - evaluating_time_start
                print("    Validation set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL_z: {:.5g}, KL_y: {:.5g}.".format(
                    ELBO_valid, ENRE_valid, KL_z_valid, KL_y_valid))
                
                print()
            
                # Plot latent validation values
                under_10 = epoch < 10
                under_100 = epoch < 100 and (epoch + 1) % 10 == 0
                under_1000 = epoch < 1000 and (epoch + 1) % 50 == 0 
                last_one = epoch == number_of_epochs - 1
                if under_10 or under_100 or under_1000 or last_one:
                    if "mixture" in self.latent_distribution_name:
                        K = self.K
                        L = self.latent_size
                        p_z_covariance_matrices = numpy.empty([K, L, L])
                        q_z_covariance_matrices = numpy.empty([K, L, L])
                        for k in range(K):
                            p_z_covariance_matrices[k] = numpy.diag(p_z_variances[k])
                            q_z_covariance_matrices[k] = numpy.diag(q_z_variances[k])
                        centroids = {
                            "prior": {
                                "probabilities": p_z_probabilities,
                                "means": numpy.stack(p_z_means),
                                "covariance_matrices": p_z_covariance_matrices
                            },
                            "posterior": {
                                "probabilities": q_z_probabilities,
                                "means": q_z_means,
                                "covariance_matrices": q_z_covariance_matrices
                            }
                        }
                    else:
                        centroids = None
                    analyseIntermediateResults(
                        z_mean_valid, validation_set, centroids, epoch,
                        self.training_name, self.main_results_directory
                    )
                    print()

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
            
            return status
    
    def evaluate(self, test_set, batch_size = 100):
        
        if self.count_sum:
            n_test = test_set.normalised_count_sum
        
        M_test = test_set.number_of_examples
        F_test = test_set.number_of_features
        
        noisy_preprocess = test_set.noisy_preprocess
        
        if not noisy_preprocess:
            
            x_test = test_set.preprocessed_values
        
            if self.reconstruction_distribution_name == "bernoulli":
                t_test = binarise(test_set.values)
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
                raise BaseError(
                    "Cannot evaluate model when it has not been trained.")
            
            evaluating_time_start = time()
            
            ELBO_test = 0
            KL_z_test = 0
            KL_y_test = 0
            ENRE_test = 0
            q_z_probabilities = numpy.zeros(self.K)
            q_z_means = numpy.zeros((self.K, self.latent_size))
            q_z_variances = numpy.zeros((self.K, self.latent_size))
            p_z_probabilities = numpy.zeros(self.K)
            p_z_means = numpy.zeros((self.K, self.latent_size))
            p_z_variances = numpy.zeros((self.K, self.latent_size))
            z_mean_test = numpy.zeros((M_test, self.latent_size))
            x_mean_test = numpy.zeros((M_test, F_test))
            y_mean_test = numpy.zeros((M_test, self.K))

            for i in range(0, M_test, batch_size):
                subset = slice(i, (i + batch_size))
                x_batch = x_test[subset]
                t_batch = t_test[subset]
                feed_dict_batch = {
                    self.x: x_batch,
                    self.t: t_batch,
                    self.is_training: False,
                    self.warm_up_weight: 1.0,
                    self.number_of_iw_samples:
                        self.number_of_importance_samples["evaluation"],
                    self.number_of_mc_samples:
                        self.number_of_monte_carlo_samples["evaluation"]
                }
                if self.count_sum:
                    feed_dict_batch[self.n] = n_test[subset]
                
                ELBO_i, ENRE_i, KL_z_i, KL_y_i, \
                q_z_probabilities_i, q_z_means_i, q_z_variances_i, p_z_probabilities_i, p_z_means_i, p_z_variances_i, x_mean_i, y_mean_i, z_mean_i = session.run(
                    [self.ELBO, self.ENRE, self.KL_z, self.KL_y, self.q_z_probabilities, self.q_z_means, self.q_z_variances, self.p_z_probabilities, self.p_z_means, self.p_z_variances, self.x_mean, self.y_mean, self.z_mean],
                    feed_dict = feed_dict_batch
                )
                
                ELBO_test += ELBO_i
                KL_z_test += KL_z_i
                KL_y_test += KL_y_i
                ENRE_test += ENRE_i
                q_z_probabilities += numpy.array(q_z_probabilities_i)
                q_z_means += numpy.array(q_z_means_i)
                q_z_variances += numpy.array(q_z_variances_i)
                p_z_probabilities += numpy.array(p_z_probabilities_i)
                p_z_means += numpy.array(p_z_means_i)
                p_z_variances += numpy.array(p_z_variances_i)
                x_mean_test[subset] = x_mean_i 
                y_mean_test[subset] = y_mean_i 
                z_mean_test[subset] = z_mean_i 
            
            ELBO_test /= M_test / batch_size
            KL_z_test /= M_test / batch_size
            KL_y_test /= M_test / batch_size
            ENRE_test /= M_test / batch_size
            q_z_probabilities /= M_test / batch_size
            q_z_means /= M_test / batch_size
            q_z_variances /= M_test / batch_size
            p_z_probabilities /= M_test / batch_size
            p_z_means /= M_test / batch_size
            p_z_variances /= M_test / batch_size
            
            summary = tf.Summary()
            summary.value.add(tag="losses/lower_bound",
                simple_value = ELBO_test)
            summary.value.add(tag="losses/reconstruction_error",
                simple_value = ENRE_test)
            summary.value.add(tag="losses/kl_divergence_z",
                simple_value = KL_z_test)
            summary.value.add(tag="losses/kl_divergence_y",
                simple_value = KL_y_test)

            for k in range(self.K):
                summary.value.add(
                    tag="prior/cluster_{}/probability".format(k),
                    simple_value = p_z_probabilities[k]
                )
                summary.value.add(
                    tag="posterior/cluster_{}/probability".format(k),
                    simple_value = q_z_probabilities[k]
                )
                for l in range(self.latent_size):
                    summary.value.add(
                        tag="prior/cluster_{}/mean/dimension_{}".format(k, l),
                        simple_value = p_z_means[k][l]
                    )
                    summary.value.add(
                        tag="posterior/cluster_{}/mean/dimension_{}".format(k, l),
                        simple_value = q_z_means[k, l]
                    )
                    summary.value.add(
                        tag="prior/cluster_{}/variance/dimension_{}"\
                            .format(k, l),
                        simple_value = p_z_variances[k][l]
                    )
                    summary.value.add(
                        tag="posterior/cluster_{}/variance/dimension_{}"\
                            .format(k, l),
                        simple_value = q_z_variances[k, l]
                    )

            test_summary_writer.add_summary(summary,
                global_step = epoch + 1)
            test_summary_writer.flush()
            
            evaluating_duration = time() - evaluating_time_start
            print("    Validation set ({}): ".format(
                formatDuration(evaluating_duration)) + \
                "ELBO: {:.5g}, ENRE: {:.5g}, KL_z: {:.5g}, KL_y: {:.5g}.".format(
                ELBO_test, ENRE_test, KL_z_test, KL_y_test))
            
            if noisy_preprocess and \
                self.reconstruction_distribution_name == "bernoulli":
                
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
            
            z_test_set = DataSet(
                name = test_set.name,
                values = z_mean_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = numpy.array(["z variable {}".format(
                    i + 1) for i in range(self.latent_size)]),
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "z"
            )

            y_test_set = DataSet(
                name = test_set.name,
                values = y_mean_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = numpy.array(["y variable {}".format(
                    i + 1) for i in range(self.K)]),
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "y"
            )
            latent_test_sets = (z_test_set, y_test_set)

            return transformed_test_set, reconstructed_test_set, latent_test_sets
