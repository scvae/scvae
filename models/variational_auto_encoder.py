import tensorflow as tf

from models.auxiliary import dense_layer

from tensorflow.python.ops.nn import relu, softmax
from tensorflow import sigmoid, identity

from tensorflow.contrib.distributions import Normal, Bernoulli, kl, Categorical
from distributions import distributions, Categorized


import numpy
from numpy import inf

import os, shutil
from time import time
from auxiliary import formatDuration

import data

class VariationalAutoEncoder(object):
    def __init__(self, feature_size, latent_size, hidden_sizes, latent_distribution = "normal",  
        number_of_latent_clusters = 2,
        reconstruction_distribution = None,
        number_of_reconstruction_classes = None,
        batch_normalisation = True, count_sum = True,
        number_of_warm_up_epochs = 0, epsilon = 1e-6,
        log_directory = "log"):
        
        print("Model setup:")
        print("    feature size: {}".format(feature_size))
        print("    latent size: {}".format(latent_size))
        print("    hidden sizes: {}".format(", ".join(map(str, hidden_sizes))))
        print("    latent distribution: " + 
            latent_distribution)
        print("    reconstruction distribution: " + reconstruction_distribution)
        if number_of_reconstruction_classes > 0:
            print("    reconstruction classes: {}".format(number_of_reconstruction_classes),
                  " (including 0s)")
        if batch_normalisation:
            print("    using batch normalisation")
        if count_sum:
            print("    using count sums")
        print("")
        
        # Class setup
        
        super(VariationalAutoEncoder, self).__init__()
        
        self.type = "VAE"
        
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        self.latent_distribution_name = latent_distribution
        self.latent_distribution = distributions[latent_distribution]
        self.number_of_latent_clusters = number_of_latent_clusters

        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = distributions[reconstruction_distribution]
        
        self.k_max = number_of_reconstruction_classes
        
        self.batch_normalisation = batch_normalisation

        self.count_sum_feature = count_sum
        self.count_sum = self.count_sum_feature or "constrained" in \
            self.reconstruction_distribution_name or "multinomial" in \
            self.reconstruction_distribution_name

        self.number_of_warm_up_epochs = number_of_warm_up_epochs

        self.epsilon = epsilon
        
        self.log_directory = os.path.join(log_directory, self.type, self.name)
        
        # Graph setup
        
        self.graph = tf.Graph()
        
        self.parameter_summary_list = []
        
        with self.graph.as_default():
            
            self.x = tf.placeholder(tf.float32, [None, self.feature_size], 'X')
            self.t = tf.placeholder(tf.float32, [None, self.feature_size], 'T')
            
            if self.count_sum:
                self.n = tf.placeholder(tf.float32, [None, 1], 'N')
            
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            
            self.warm_up_weight = tf.placeholder(tf.float32, [], 'warm_up_weight')
            parameter_summary = tf.summary.scalar('warm_up_weight',
                self.warm_up_weight)
            self.parameter_summary_list.append(parameter_summary)
            
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')
            
            self.inference()
            self.loss()
            self.training()
            
            self.saver = tf.train.Saver(max_to_keep = 1)
            
            print("Trainable parameters:")
        
            trainable_parameters = tf.trainable_variables()
        
            width = max(map(len, [p.name for p in trainable_parameters]))
        
            for parameter in trainable_parameters:
                print("    {:{}}  {}".format(
                    parameter.name, width, parameter.get_shape()))
    
    @property
    def name(self):
        
        model_name = self.reconstruction_distribution_name.replace(" ", "_")
        
        if self.k_max:
            model_name += "_c_" + str(self.k_max)
        
        if self.count_sum_feature:
            model_name += "_sum"
        
        model_name += "_l_" + str(self.latent_size) \
            + "_h_" + "_".join(map(str, self.hidden_sizes))
        
        if self.batch_normalisation:
            model_name += "_bn"
        
        if self.number_of_warm_up_epochs:
            model_name += "_wu_" + str(self.number_of_warm_up_epochs)
        
        return model_name
    
    @property
    def description(self):
        
        description = "VAE"
        
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
        
        description += " (" + ", ".join(configuration) + ")"
        
        return description
    
    def inference(self):
        
        encoder = self.x
        
        with tf.variable_scope("ENCODER"):
            for i, hidden_size in enumerate(self.hidden_sizes):
                encoder = dense_layer(
                    inputs = encoder,
                    num_outputs = hidden_size,
                    activation_fn = relu,
                    batch_normalisation = self.batch_normalisation, 
                    is_training = self.is_training,
                    scope = '{:d}'.format(i + 1)
                )
        
        with tf.variable_scope("Z"):
            z_phi = {}
        
            # Retrieving layers for all latent distribution parameters
            for parameter in self.latent_distribution["parameters"]:
                
                parameter_activation_function = \
                    self.latent_distribution["parameters"]\
                    [parameter]["activation function"]
                p_min, p_max = \
                    self.latent_distribution["parameters"]\
                    [parameter]["support"]
                
                if self.latent_distribution_name == "gaussian mixture":
                    print("Setting up Gaussian mixture posterior")
                    print("Setting up " + parameter)
                    if parameter == "mus" or parameter == "log_sigmas":
                        z_phi[parameter] = []
                        for k in range(self.number_of_latent_clusters):
                            z_phi[parameter].append(
                                dense_layer(
                                    inputs = encoder,
                                    num_outputs = self.latent_size,
                                    activation_fn = lambda x: tf.clip_by_value(
                                        parameter_activation_function(x),
                                        p_min + self.epsilon,
                                        p_max - self.epsilon
                                    ),
                                    is_training = self.is_training,
                                    scope = parameter.upper()[:-1] + str(k)
                                )
                            )
                        print("Number of " + parameter + " in gaussian mixture:", len(z_phi[parameter]))
                    else:
                        z_phi[parameter] = dense_layer(
                            inputs = encoder,
                            num_outputs = self.number_of_latent_clusters,
                            activation_fn = lambda x: tf.clip_by_value(
                                parameter_activation_function(x),
                                p_min + self.epsilon,
                                p_max - self.epsilon
                            ),
                            is_training = self.is_training,
                            scope = parameter.upper()
                        )
                else:
                    z_phi[parameter] = dense_layer(
                        inputs = encoder,
                        num_outputs = self.latent_size,
                        activation_fn = lambda x: tf.clip_by_value(
                            parameter_activation_function(x),
                            p_min + self.epsilon,
                            p_max - self.epsilon
                        ),
                        is_training = self.is_training,
                        scope = parameter.upper()
                    )

            # Parameterise latent distribution 
            self.q_z_given_x = self.latent_distribution["class"](z_phi)

            # Mean of latent distribution
            self.z_mean = self.q_z_given_x.mean()
        
            # Stochastic layer sampling with reparameterisation trick
            self.z = tf.cast(self.q_z_given_x.sample(), tf.float32)
        

        
        # Decoder - Generative model, p(x|z)
        
        if self.count_sum_feature:
            decoder = tf.concat([self.z, self.n], axis = 1, name = 'Z_N')
        else:
            decoder = self.z
        
        with tf.variable_scope("DECODER"):
            for i, hidden_size in enumerate(reversed(self.hidden_sizes)):
                decoder = dense_layer(
                    inputs = decoder,
                    num_outputs = hidden_size,
                    activation_fn = relu,
                    batch_normalisation = self.batch_normalisation,
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
            
            if "constrained" in self.reconstruction_distribution_name or \
                "multinomial" in self.reconstruction_distribution_name:
                self.p_x_given_z = self.reconstruction_distribution["class"](x_theta, self.n)
            elif "multinomial" in self.reconstruction_distribution_name:
               self.p_x_given_z = self.reconstruction_distribution["class"](x_theta, self.n) 
            else:
                self.p_x_given_z = self.reconstruction_distribution["class"](x_theta)
        
            if self.k_max:
                
                x_logits = dense_layer(
                    inputs = decoder,
                    num_outputs = self.feature_size * self.k_max,
                    activation_fn = None,
                    is_training = self.is_training,
                    scope = "P_K"
                )
                
                x_logits = tf.reshape(x_logits,
                    [-1, self.feature_size, self.k_max])
                
                self.p_x_given_z = Categorized(
                    dist = self.p_x_given_z,
                    cat = Categorical(logits = x_logits)
                )
            
            self.x_tilde_mean = self.p_x_given_z.mean()
        
        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            parameter_summary = tf.summary.histogram(parameter.name, parameter)
            self.parameter_summary_list.append(parameter_summary)
        self.parameter_summary = tf.summary.merge(self.parameter_summary_list)
    
    def loss(self):
        
        # Recognition prior
        if self.latent_distribution_name == "normal":
            p_z_mu = tf.constant(0.0, dtype = tf.float32)
            p_z_sigma = tf.constant(1.0, dtype = tf.float32)
            p_z = Normal(p_z_mu, p_z_sigma)
        elif self.latent_distribution_name == "bernoulli":
            p_z_p = tf.constant(0.0, dtype = tf.float32)
            p_z = Bernoulli(p = p_z_p)
        
        # Loss
        
        ## Reconstruction error
        log_p_x_given_z = tf.reduce_mean(
            tf.reduce_sum(self.p_x_given_z.log_prob(self.t), axis = 1),
            name = 'reconstruction_error'
        )
        tf.add_to_collection('losses', log_p_x_given_z)
        self.ENRE = log_p_x_given_z
        
        if self.latent_distribution_name == "gaussian mixture":
            KL_qp_all = tf.reduce_mean(self.q_z_given_x.entropy_lower_bound(), axis = 0)
        else:
            KL_qp_all = tf.reduce_mean(kl(self.q_z_given_x, p_z), axis = 0)
        
        self.KL_all = KL_qp_all

        ## Regularisation
        KL_qp = tf.reduce_sum(KL_qp_all, name = "kl_divergence")
        tf.add_to_collection('losses', KL_qp)
        self.KL = KL_qp
        
        # Averaging over samples.
        self.lower_bound = tf.subtract(log_p_x_given_z, 
            tf.where(self.is_training, self.warm_up_weight * KL_qp, KL_qp), name = 'lower_bound')
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
        reset_training = False):
        
        # Logging
        
        # parameter_values = "lr_{:.1g}".format(learning_rate)
        # parameter_values += "_b_" + str(batch_size)
        
        # self.log_directory = os.path.join(self.log_directory, parameter_values)
        
        if reset_training and os.path.exists(self.log_directory):
            shutil.rmtree(self.log_directory)
        
        checkpoint_file = os.path.join(self.log_directory, 'model.ckpt')
        
        # Setup
        
        if self.count_sum:
            n_train = training_set.counts.sum(axis = 1).reshape(-1, 1)
            n_valid = validation_set.counts.sum(axis = 1).reshape(-1, 1)
        
        M_train = training_set.number_of_examples
        M_valid = validation_set.number_of_examples
        
        x_train = training_set.preprocessed_counts
        x_valid = validation_set.preprocessed_counts
        
        t_train = training_set.counts
        t_valid = validation_set.counts
        
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
                        self.warm_up_weight: warm_up_weight
                    }
                    
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[batch_indices]

                    # print(session.run([
                    #     tf.reduce_sum(self.p_x_given_z.dist.xi), 
                    #     tf.reduce_sum(self.p_x_given_z.dist.sigma), 
                    #     tf.reduce_sum(self.q_z_given_x.mu), 
                    #     tf.reduce_sum(self.q_z_given_x.sigma)
                    #     ], 
                    #     feed_dict=feed_dict_batch)
                    # )
                    
                    # Run the stochastic batch training operation
                    _, batch_loss = session.run(
                        [self.train_op, self.lower_bound],
                        feed_dict = feed_dict_batch
                    )

                    # print(session.run([
                    #     tf.reduce_sum(self.p_x_given_z.dist.xi), 
                    #     tf.reduce_sum(self.p_x_given_z.dist.sigma), 
                    #     tf.reduce_sum(self.q_z_given_x.mu), 
                    #     tf.reduce_sum(self.q_z_given_x.sigma)
                    #     ], 
                    #     feed_dict=feed_dict_batch)
                    # )

                    # Compute step duration
                    step_duration = time() - step_time_start
                    
                    # Print evaluation and output summaries
                    if (step + 1 - steps_per_epoch * epoch) in output_at_step:

                        print('Step {:d} ({}): {:.5g}.'.format(
                            int(step + 1), formatDuration(step_duration),
                            batch_loss))
                                    
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
                KL_train = 0
                ENRE_train = 0
                
                z_KL = numpy.zeros(self.latent_size)
                
                for i in range(0, M_train, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_train[subset]
                    t_batch = x_train[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.warm_up_weight: warm_up_weight
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[subset]
                    
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
                
                evaluating_duration = time() - evaluating_time_start
                
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_train)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_train)
                summary.value.add(tag="losses/kl_divergence",
                    simple_value = KL_train)
                
                for i in range(self.latent_size):
                    summary.value.add(tag="kl_divergence_neurons/{}".format(i),
                        simple_value = z_KL[i])
                
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
                
                for i in range(0, M_valid, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_valid[subset]
                    t_batch = t_valid[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                        self.warm_up_weight: warm_up_weight
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_valid[subset]
                    
                    ELBO_i, KL_i, ENRE_i = session.run(
                        [self.ELBO, self.KL, self.ENRE],
                        feed_dict = feed_dict_batch
                    )
                    
                    ELBO_valid += ELBO_i
                    KL_valid += KL_i
                    ENRE_valid += ENRE_i
                
                ELBO_valid /= M_valid / batch_size
                KL_valid /= M_valid / batch_size
                ENRE_valid /= M_valid / batch_size
                
                summary = tf.Summary()
                summary.value.add(tag="losses/lower_bound",
                    simple_value = ELBO_valid)
                summary.value.add(tag="losses/reconstruction_error",
                    simple_value = ENRE_valid)
                summary.value.add(tag="losses/kl_divergence",
                    simple_value = KL_valid)
                validation_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                validation_summary_writer.flush()
                
                evaluating_duration = time() - evaluating_time_start
                print("    Validation set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                    ELBO_valid, ENRE_valid, KL_valid))
                
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
                
    
    def evaluate(self, test_set, batch_size = 100):
        
        if self.count_sum:
            n_test = test_set.counts.sum(axis = 1).reshape(-1, 1)
        
        M_test = test_set.number_of_examples
        F_test = test_set.number_of_features
        
        x_test = test_set.preprocessed_counts
        
        t_test = test_set.counts
        
        checkpoint = tf.train.get_checkpoint_state(self.log_directory)
        
        with tf.Session(graph = self.graph) as session:
        
            if checkpoint:
                self.saver.restore(session, checkpoint.model_checkpoint_path)
            
            evaluating_time_start = time()
            
            ELBO_test = 0
            KL_test = 0
            ENRE_test = 0
            
            x_tilde_test = numpy.empty([M_test, F_test])
            z_mean_test = numpy.empty([M_test, self.latent_size])
            
            for i in range(0, M_test, batch_size):
                subset = slice(i, (i + batch_size))
                x_batch = x_test[subset]
                t_batch = t_test[subset]
                feed_dict_batch = {
                    self.x: x_batch,
                    self.t: t_batch,
                    self.is_training: False,
                    self.warm_up_weight: 1.0
                }
                if self.count_sum:
                    feed_dict_batch[self.n] = n_test[subset]
                
                ELBO_i, KL_i, ENRE_i, x_tilde_i, z_mean_i = session.run(
                    [self.ELBO, self.KL, self.ENRE,
                        self.x_tilde_mean, self.z_mean],
                    feed_dict = feed_dict_batch
                )
                
                ELBO_test += ELBO_i
                KL_test += KL_i
                ENRE_test += ENRE_i
                
                x_tilde_test[subset] = x_tilde_i
                z_mean_test[subset] = z_mean_i
            
            ELBO_test /= M_test / batch_size
            KL_test /= M_test / batch_size
            ENRE_test /= M_test / batch_size
            
            evaluating_duration = time() - evaluating_time_start
            print("Test set ({}): ".format(
                formatDuration(evaluating_duration)) + \
                "ELBO: {:.5g}, ENRE: {:.5g}, KL: {:.5g}.".format(
                ELBO_test, ENRE_test, KL_test))
            
            evaluation_test = {
                "ELBO": ELBO_test,
                "ENRE": ENRE_test,
                "KL": KL_test
            }
            
            reconstructed_test_set = data.BaseDataSet(
                counts = x_tilde_test,
                cells = test_set.cells,
                genes = test_set.genes,
                name = self.name,
                kind = "test",
                version = "reconstructed"
            )
            
            return reconstructed_test_set, z_mean_test, evaluation_test
