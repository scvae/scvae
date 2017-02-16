import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.python.ops.nn import relu
from tensorflow import sigmoid
from tensorflow.contrib.distributions import Bernoulli, Normal, Poisson

import numpy

from time import time

class VariationalAutoEncoder(object):
    def __init__(self, feature_size, latent_size, hidden_sizes,
        reconstruction_distribution = None, number_of_reconstruction_classes = None,
        use_batch_norm = False):
    
        # Setup
    
        super(VariationalAutoEncoder, self).__init__()
    
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        
        self.reconstruction_distribution = reconstruction_distribution
        
        self.use_batch_norm = use_batch_norm
    
        
    
    def name(self):
        pass
        # model_name = base_name + "_" + \
        #     dataSetBaseName(splitting_method, splitting_fraction,
        #     filtering_method, feature_selection, feature_size)
        #
        # model_name += "_r_" + reconstruction_distribution.replace(" ", "_")
        #
        # if number_of_reconstruction_classes:
        #     model_name += "_c_" + str(reconstruction_classes)
        #
        # if use_count_sum:
        #     model_name += "_sum"
        #
        # model_name += "_l_" + str(latent_size) + "_h_" + "_".join(map(str,
        #     hidden_structure))
        #
        # if use_batch_norm:
        #     model_name += "_bn"
        #
        # model_name += "_lr_{:.1g}".format(learning_rate)
        # model_name += "_b_" + str(batch_size) + "_wu_" + str(number_of_warm_up_epochs)
        #
        # if use_gpu:
        #     model_name += "_gpu"
        #
        # model_name += "_e_" + str(number_of_epochs)
        #
        # return model_name
    
    def inference(self):
        # initialize placeholders, symbolics, with shape (batchsize, features)
        
        batch_norm_decay = 0.999
        
        l_enc = self.x
        # Encoder - Recognition Model, q(z|x)
        for i, hidden_size in enumerate(self.hidden_sizes):
            l_enc = dense_layer(inputs=l_enc, num_outputs=hidden_size, activation_fn=relu, use_batch_norm=self.use_batch_norm, decay=batch_norm_decay,is_training=self.phase, scope='ENCODER{:d}'.format(i + 1))

        self.l_mu_z = dense_layer(inputs=l_enc, num_outputs=self.latent_size, activation_fn=None, use_batch_norm=False, is_training=self.phase, scope='ENCODER_MU_Z')
        self.l_logvar_z = tf.clip_by_value(dense_layer(inputs=l_enc, num_outputs=self.latent_size, activation_fn=None, use_batch_norm=False, is_training=self.phase, scope='ENCODER_LOGVAR_Z'), -10, 10)

        # Stochastic layer
        ## Sample latent variable: z = mu + sigma*epsilon
        l_z = sample_layer(self.l_mu_z, self.l_logvar_z, 'SAMPLE_LAYER')

        # Decoder - Generative model, p(x|z)
        l_dec = l_z
        for i, hidden_size in enumerate(reversed(self.hidden_sizes)):
            l_dec = dense_layer(inputs=l_dec, num_outputs=hidden_size, activation_fn=relu, use_batch_norm=self.use_batch_norm, decay=batch_norm_decay, is_training=self.phase, scope='DECODER{:d}'.format(i + 1))

        # Reconstruction Distribution Parameterization
        if self.reconstruction_distribution == 'bernoulli':
            l_dec_out_p = dense_layer(inputs=l_dec, num_outputs=self.feature_size, activation_fn=sigmoid, use_batch_norm=False, is_training=self.phase, scale=True, scope='DECODER_BERNOULLI_P')
            self.recon_dist = Bernoulli(p = l_dec_out_p)

        elif self.reconstruction_distribution == 'normal':
            l_dec_out_mu = dense_layer(inputs=l_dec, num_outputs=self.feature_size, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=self.phase, scope='DECODER_NORMAL_MU')
            l_dec_out_log_sigma = dense_layer(inputs=l_dec, num_outputs=self.feature_size, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=self.phase, scope='DECODER_NORMAL_LOG_SIGMA')
            self.recon_dist = Normal(mu=l_dec_out_mu, 
                sigma=tf.exp(tf.clip_by_value(l_dec_out_log_sigma, -3, 3)))

        elif self.reconstruction_distribution == 'poisson':
            l_dec_out_log_lambda = dense_layer(inputs=l_dec, num_outputs=self.feature_size, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=self.phase, scope='DECODER_POISSON_LOG_LAMBDA')
            self.recon_dist = Poisson(lam=tf.exp(tf.clip_by_value(l_dec_out_log_lambda, -10, 10)))
    
    def loss(self):
        # Loss
        # Reconstruction error. (all log(p) are in [-\infty, 0]). 
        log_px_given_z = tf.reduce_mean(tf.reduce_sum(self.recon_dist.log_prob(self.x), axis = 1), name='reconstruction_error')
        # Regularization: Kulback-Leibler divergence between approximate posterior, q(z|x), and isotropic gauss prior p(z)=N(z,mu,sigma*I).
        KL_qp = tf.reduce_mean(kl_normal2_stdnormal(self.l_mu_z, self.l_logvar_z, eps=1e-6), name='kl_divergence')

        # Averaging over samples.  
        self.loss_op = tf.subtract(log_px_given_z, KL_qp, name='lower_bound')
        
        tf.add_to_collection('losses', KL_qp)
        tf.add_to_collection('losses', log_px_given_z)
    
    def training(self):
        """Sets up the training Ops.

        Creates a summarizer to track the loss over time in TensorBoard.

        Creates an optimizer and applies the gradients to all trainable variables.

        The Op returned by this function is what must be passed to the
        `session.run()` call to cause the model to train.

        Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

        Returns:
        train_op: The Op for training.
        """
        # Add a scalar summary for the snapshot loss.
        for l in tf.get_collection('losses') + [self.loss_op]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name, l)
        for parameter in tf.trainable_variables():
            tf.summary.histogram(parameter.name, parameter)
        # Create the gradient descent optimizer with the given learning rate.
        # Make sure that the Updates of the moving_averages in batch_norm layers
        # are performed before the train_step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                # Optimizer and training objective of negative loss
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Create a variable to track the global step.
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
            
                # Use the optimizer to apply the gradients that minimize the loss
                # (and also increment the global step counter) as a single training step.
                self.train_op = optimizer.minimize(-self.loss_op, global_step=self.global_step)
        else:
            # Optimizer and training objective of negative loss
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            # Create a variable to track the global step.
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
            # Use the optimizer to apply the gradients that minimize the loss
            # (and also increment the global step counter) as a single training step.
            self.train_op = optimizer.minimize(-self.loss_op, global_step=self.global_step)
        
    
    def train(self, train_data, valid_data, number_of_epochs=50, batch_size=100, learning_rate=1e-3, log_directory = None, reset_training = False):
        
        with tf.Graph().as_default():
            
            self.x = tf.placeholder(tf.float32, [None, self.feature_size], 'x') # counts
    
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            self.warm_up_weight = tf.placeholder(tf.float32, [], 'warm_up_weight')
    
            self.phase = tf.placeholder(tf.bool, [], 'phase')
            
            self.inference()
            self.loss()
            self.training()
        
            self.summary = tf.summary.merge_all()
        
            for parameter in tf.trainable_variables():
                print(parameter.name, parameter.get_shape())
            
            # Train
            M = train_data.number_of_examples
        
            saver = tf.train.Saver()
        
            session = tf.Session()
        
            summary_writer = tf.summary.FileWriter(log_directory, session.graph)
        
            session.run(tf.global_variables_initializer())
        
            #train_losses, valid_losses = [], []
            feed_dict_train = {self.x: train_data.counts, self.phase: False}
            feed_dict_valid = {self.x: valid_data.counts, self.phase: False}
            for epoch in range(number_of_epochs):
                shuffled_indices = numpy.random.permutation(M)
                for i in range(0, M, batch_size):
                
                    step = i / batch_size
                
                    start_time = time()
                
                    subset = shuffled_indices[i:(i + batch_size)]
                    batch = train_data.counts[subset]
                    feed_dict = {self.x: batch, self.phase: True, self.learning_rate: learning_rate}
                    _, batch_loss = session.run([self.train_op, self.loss_op], feed_dict=feed_dict)
                
                    duration = time() - start_time
                
                    if step % 10:
                        print('Step {:d}: loss = {:.2f} ({:.3f} sec)'.format(int(step), batch_loss, duration))
                        summary_str = session.run(self.summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                    
                train_loss = session.run(self.loss_op, feed_dict=feed_dict_train)
                valid_loss = session.run(self.loss_op, feed_dict=feed_dict_valid)
                
                checkpoint_file = os.path.join(log_directory, 'model.ckpt')
                saver.save(session, checkpoint_file, global_step=step)
                
                print("Epoch %d: ELBO: %g (Train), %g (Valid)"%(epoch+1, train_loss, valid_loss))
                



# Layer for sampling latent variables using the reparameterization trick 
def sample_layer(mean, log_var, scope='sample_layer'):
    with tf.variable_scope(scope):
        input_shape  = tf.shape(mean)
        batch_size = input_shape[0]
        num_latent = input_shape[1]
        eps = tf.random_normal((batch_size, num_latent), 0, 1, dtype=tf.float32)
        # Sample z = mu + sigma*epsilon
        return mean + tf.exp(0.5 * log_var) * eps


# Wrapper layer for inserting batch normalization in between linear and nonlinear activation layers. 
def dense_layer(inputs, num_outputs, is_training, scope, activation_fn=None, use_batch_norm=False, decay=0.999, center=True, scale=False):
    with tf.variable_scope(scope):
        outputs = fully_connected(inputs, num_outputs=num_outputs, activation_fn=None, scope=scope+'/DENSE')
        if use_batch_norm:
            outputs = batch_norm(outputs, center=center, scale=scale, is_training=is_training, scope='BATCH_NORM')
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def kl_normal2_stdnormal(mean, log_var, eps=0.0):
    """
    Compute analytically integrated KL-divergence between a diagonal covariance Gaussian and 
    a standard Gaussian.

    In the setting of the variational autoencoder, when a Gaussian prior and diagonal Gaussian 
    approximate posterior is used, this analytically integrated KL-divergence term yields a lower variance 
    estimate of the likelihood lower bound compared to computing the term by Monte Carlo approximation.

        .. math:: D_{KL}[q_{\phi}(z|x) || p_{\theta}(z)]

    See appendix B of [KINGMA]_ for details.

    Parameters
    ----------
    mean : Tensorflow tensor
        Mean of the diagonal covariance Gaussian.
    log_var : Tensorflow tensor
        Log variance of the diagonal covariance Gaussian.

    Returns
    -------
    Tensorflow tensor
        Element-wise KL-divergence, this has to be summed when the Gaussian distributions are multi-variate.

    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).

    """
    return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
