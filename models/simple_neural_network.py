import tensorflow as tf

from models.auxiliary import dense_layer

from tensorflow.python.ops.nn import relu, softmax
from tensorflow import sigmoid, identity

from tensorflow.contrib.distributions import Normal, kl, Categorical
from distributions import distributions, Categorized
import numpy
from numpy import inf

import os, shutil
from time import time
from auxiliary import formatDuration, normaliseString

from data import DataSet, binarise

class SimpleNeuralNetwork(object):
    def __init__(self, feature_size, hidden_sizes,
        reconstruction_distribution = None,
        number_of_reconstruction_classes = None,
        batch_normalisation = True, count_sum = True,
        epsilon = 1e-6,
        log_directory = "log"):
        
        # Class setup
        
        super(SimpleNeuralNetwork, self).__init__()
        
        self.type = "SNN"
        
        self.feature_size = feature_size
        self.hidden_sizes = hidden_sizes
        
        self.reconstruction_distribution_name = reconstruction_distribution
        self.reconstruction_distribution = distributions[reconstruction_distribution]
        
        self.k_max = number_of_reconstruction_classes
        
        self.batch_normalisation = batch_normalisation

        self.count_sum_feature = count_sum
        self.count_sum = self.count_sum_feature or "constrained" in \
            self.reconstruction_distribution_name or "multinomial" in \
            self.reconstruction_distribution_name

        self.epsilon = epsilon
        
        self.main_log_directory = log_directory
        
        # Graph setup
        
        self.graph = tf.Graph()
        
        self.parameter_summary_list = []
        
        with self.graph.as_default():
            
            self.x = tf.placeholder(tf.float32, [None, self.feature_size], 'X')
            self.t = tf.placeholder(tf.float32, [None, self.feature_size], 'T')
            
            if self.count_sum:
                self.n = tf.placeholder(tf.float32, [None, 1], 'N')
            
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')
            
            self.inference()
            self.loss()
            self.training()
            
            self.saver = tf.train.Saver(max_to_keep = 1)
    
    @property
    def name(self):
        
        model_name = self.reconstruction_distribution_name.replace(" ", "_")
        
        if self.k_max:
            model_name += "_c_" + str(self.k_max)
        
        if self.count_sum_feature:
            model_name += "_sum"
        
        model_name += "_h_" + "_".join(map(str, self.hidden_sizes))
        
        if self.batch_normalisation:
            model_name += "_bn"
        
        return model_name
        
        reconstruction_part = normaliseString(
            self.reconstruction_distribution_name)
        
        if self.k_max:
            reconstruction_part += "_c_" + str(self.k_max)
        
        if self.count_sum_feature:
            reconstruction_part += "_sum"
        
        reconstruction_part += "_h_" + "_".join(map(str, self.hidden_sizes))
        
        if self.batch_normalisation:
            reconstruction_part += "_bn"
        
        model_name = os.path.join(self.type, reconstruction_part)
        
        return model_name
    
    @property
    def log_directory(self):
        return os.path.join(self.main_log_directory, self.name)
    
    @property
    def title(self):
        
        title = model.type
        
        configuration = [
            self.reconstruction_distribution_name.capitalize(),
            "$h = \\{{{}\\}}$".format(", ".join(map(str, self.hidden_sizes)))
        ]
        
        if self.k_max:
            configuration.append("$k_{{\\mathrm{{max}}}} = {}$".format(self.k_max))
        
        if self.count_sum_feature:
            configuration.append("CS")
        
        if self.batch_normalisation:
            configuration.append("BN")
        
        title += " (" + ", ".join(configuration) + ")"
        
        return title
    
    @property
    def description(self):
        
        description_parts = ["Model setup:"]
        
        description_parts.append("type: {}".format(self.type))
        description_parts.append("feature size: {}".format(self.feature_size))
        description_parts.append("hidden sizes: {}".format(", ".join(
            map(str, self.hidden_sizes))))
        
        description_parts.append("reconstruction distribution: " +
            self.reconstruction_distribution_name)
        if self.k_max > 0:
            description_parts.append(
                "reconstruction classes: {}".format(self.k_max) +
                " (including 0s)"
            )
        
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
    
    def inference(self):
        
        if self.count_sum_feature:
            decoder = tf.concat([self.x, self.n], axis = 1, name = 'X_N')
        else:
            decoder = self.x
        
        with tf.variable_scope("NEURAL_NETWORK"):
            for i, hidden_size in enumerate(reversed(self.hidden_sizes)):
                neural_network = dense_layer(
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
                    inputs = neural_network,
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
                self.p_x = self.reconstruction_distribution["class"](x_theta, self.n)
            elif "multinomial" in self.reconstruction_distribution_name:
               self.p_x = self.reconstruction_distribution["class"](x_theta, self.n) 
            else:
                self.p_x = self.reconstruction_distribution["class"](x_theta)
        
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
                
                self.p_x = Categorized(
                    dist = self.p_x,
                    cat = Categorical(logits = x_logits)
                )
            
            self.x_tilde_mean = self.p_x.mean()
        
        # Add histogram summaries for the trainable parameters
        for parameter in tf.trainable_variables():
            parameter_summary = tf.summary.histogram(parameter.name, parameter)
            self.parameter_summary_list.append(parameter_summary)
        self.parameter_summary = tf.summary.merge(self.parameter_summary_list)
    
    def loss(self):
        
        self.log_likelihood = tf.reduce_mean(
            tf.reduce_sum(self.p_x.log_prob(self.t), axis = 1),
            name = 'log_likelihood'
        )
        tf.add_to_collection('losses', self.log_likelihood)
    
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
                -self.log_likelihood,
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
            n_train = training_set.count_sum
            n_valid = validation_set.count_sum
        
        M_train = training_set.number_of_examples
        M_valid = validation_set.number_of_examples
        
        x_train = training_set.preprocessed_values
        x_valid = validation_set.preprocessed_values
        
        if self.reconstruction_distribution_name == "bernoulli":
            t_train = binarise(training_set.values)
            t_valid = binarise(validation_set.values)
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
                    }
                    
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[batch_indices]
                    
                    # Run the stochastic batch training operation
                    _, batch_loss = session.run(
                        [self.train_op, self.log_likelihood],
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

                # Saving model parameters
                print('    Saving model.')
                saving_time_start = time()
                self.saver.save(session, checkpoint_file,
                    global_step = epoch + 1)
                saving_duration = time() - saving_time_start
                print('    Model saved ({}).'.format(
                    formatDuration(saving_duration)))
                
                # Export parameter summaries
                parameter_summary_string = session.run(self.parameter_summary)
                parameter_summary_writer.add_summary(
                    parameter_summary_string, global_step = epoch + 1)
                parameter_summary_writer.flush()
                
                # Evaluation
                print('    Evaluating model.')
                
                ## Training
                
                evaluating_time_start = time()
                
                log_likelihood_train = 0
                
                for i in range(0, M_train, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_train[subset]
                    t_batch = t_train[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_train[subset]
                    
                    log_likelihood_i = session.run(
                        self.log_likelihood,
                        feed_dict = feed_dict_batch
                    )
                    
                    log_likelihood_train += log_likelihood_i
                
                log_likelihood_train /= M_train / batch_size
                
                evaluating_duration = time() - evaluating_time_start
                
                summary = tf.Summary()
                summary.value.add(tag = "losses/log_likelihood",
                    simple_value = log_likelihood_train)
                
                training_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                training_summary_writer.flush()
                
                print("    Training set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "log-likelihood: {:.5g}.".format(log_likelihood_train))
                
                ## Validation
                
                evaluating_time_start = time()
                
                log_likelihood_valid = 0
                
                for i in range(0, M_valid, batch_size):
                    subset = slice(i, (i + batch_size))
                    x_batch = x_valid[subset]
                    t_batch = t_valid[subset]
                    feed_dict_batch = {
                        self.x: x_batch,
                        self.t: t_batch,
                        self.is_training: False,
                    }
                    if self.count_sum:
                        feed_dict_batch[self.n] = n_valid[subset]
                    
                    log_likelihood_i = session.run(
                        self.log_likelihood,
                        feed_dict = feed_dict_batch
                    )
                    
                    log_likelihood_valid += log_likelihood_i
                
                log_likelihood_valid /= M_valid / batch_size
                
                evaluating_duration = time() - evaluating_time_start
                
                summary = tf.Summary()
                summary.value.add(tag="losses/log_likelihood",
                    simple_value = log_likelihood_valid)
                
                validation_summary_writer.add_summary(summary,
                    global_step = epoch + 1)
                validation_summary_writer.flush()
                
                print("    Validation set ({}): ".format(
                    formatDuration(evaluating_duration)) + \
                    "log-likelihood: {:.5g}.".format(log_likelihood_valid))
                
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
            n_test = test_set.count_sum
        
        M_test = test_set.number_of_examples
        F_test = test_set.number_of_features
        
        x_test = test_set.preprocessed_values
        
        if self.reconstruction_distribution_name == "bernoulli":
            t_test = binarise(test_set.values)
        else:
            t_test = test_set.values
        
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
            
            log_likelihood_test = 0
            
            x_tilde_test = numpy.empty([M_test, F_test])
            
            for i in range(0, M_test, batch_size):
                subset = slice(i, (i + batch_size))
                x_batch = x_test[subset]
                t_batch = t_test[subset]
                feed_dict_batch = {
                    self.x: x_batch,
                    self.t: t_batch,
                    self.is_training: False,
                }
                if self.count_sum:
                    feed_dict_batch[self.n] = n_test[subset]
                
                log_likelihood_i, x_tilde_i = session.run(
                    [self.log_likelihood, self.x_tilde_mean],
                    feed_dict = feed_dict_batch
                )
                
                log_likelihood_test += log_likelihood_i
                
                x_tilde_test[subset] = x_tilde_i
            
            log_likelihood_test /= M_test / batch_size
            
            summary = tf.Summary()
            summary.value.add(tag="losses/log_likelihood",
                simple_value = log_likelihood_test)
            test_summary_writer.add_summary(summary,
                global_step = epoch)
            test_summary_writer.flush()
            
            evaluating_duration = time() - evaluating_time_start
            print("Test set ({}): ".format(
                formatDuration(evaluating_duration)) + \
                "log-likelihood: {:.5g}".format(log_likelihood_test))
            
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
                values = x_tilde_test,
                preprocessed_values = None,
                labels = test_set.labels,
                example_names = test_set.example_names,
                feature_names = test_set.feature_names,
                feature_selection = test_set.feature_selection,
                preprocessing_methods = test_set.preprocessing_methods,
                kind = "test",
                version = "reconstructed"
            )
            
            return transformed_test_set, reconstructed_test_set
