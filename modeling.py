"""Builds the Variational Autoencoder framework for Single Cell gene expression counts.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make reconstructions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

"""
# Model specifications
N_epochs = 21
NUM_CLASSES = 2
N_train = mnist.train.images.shape[0]
print(NUM_FEAT)
N_z = 2
BATCH_SIZE = 100
N_batches = N_train // BATCH_SIZE
reconstruction_distribution_name = 'poisson'
hidden_structure = [64, 32, 16]
use_batch_norm = True
plot_manifold = True


def inference(inputs, hidden_sizes, latent_size, feature_size, reconstruction_distribution):
	# initialize placeholders, symbolics, with shape (batchsize, features)
	#x_in = tf.placeholder(tf.float32, [None, NUM_FEAT], 'x_in')
	phase = tf.placeholder(tf.bool, [], 'phase')

	l_enc = inputs
	# Encoder - Recognition Model, q(z|x)
	for i, hidden_size in enumerate(hidden_sizes):
	 	l_enc = dense_layer(inputs=l_enc, num_outputs=hidden_size, activation_fn=relu, use_batch_norm=use_batch_norm, decay=batch_norm_decay,is_training=phase, scope='ENCODER{:d}'.format(i + 1))

	l_mu_z = dense_layer(inputs=l_enc, num_outputs=latent_size, activation_fn=None, use_batch_norm=False, is_training=phase, scope='ENCODER_MU_Z')
	l_logvar_z = tf.clip_by_value(dense_layer(inputs=l_enc, num_outputs=latent_size, activation_fn=None, use_batch_norm=False, is_training=phase, scope='ENCODER_LOGVAR_Z'), -10, 10)

	# Stochastic layer
	## Sample latent variable: z = mu + sigma*epsilon
	l_z = sample_layer(l_mu_z, l_logvar_z, 'SAMPLE_LAYER')

	# Decoder - Generative model, p(x|z)
	l_dec = l_z
	for i, hidden_size in enumerate(reversed(hidden_sizes)):
		l_dec = dense_layer(inputs=l_dec, num_outputs=hidden_size, activation_fn=relu, use_batch_norm=use_batch_norm, decay=batch_norm_decay, is_training=phase, scope='DECODER{:d}'.format(i + 1))

	# Reconstruction Distribution Parameterization
	if reconstruction_distribution_name == 'bernoulli':
		l_dec_out_p = dense_layer(inputs=l_dec, num_outputs=feature_size, activation_fn=sigmoid, use_batch_norm=False, is_training=phase, scale=True, scope='DECODER_BERNOULLI_P')
		recon_dist = Bernoulli(p = l_dec_out_p)

	elif reconstruction_distribution_name == 'normal':
		l_dec_out_mu = dense_layer(inputs=l_dec, num_outputs=feature_size, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=phase, scope='DECODER_NORMAL_MU')
		l_dec_out_log_sigma = dense_layer(inputs=l_dec, num_outputs=feature_size, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=phase, scope='DECODER_NORMAL_LOG_SIGMA')
		recon_dist = Normal(mu=l_dec_out_mu, 
			sigma=tf.exp(tf.clip_by_value(l_dec_out_log_sigma, -3, 3)))

	elif reconstruction_distribution_name == 'poisson':
		l_dec_out_log_lambda = dense_layer(inputs=l_dec, num_outputs=feature_size, activation_fn=None, use_batch_norm=False, decay=batch_norm_decay, is_training=phase, scope='DECODER_POISSON_LOG_LAMBDA')
		recon_dist = Poisson(lam=tf.exp(tf.clip_by_value(l_dec_out_log_lambda, -10, 10)))

	return l_mu_z, l_logvar_z, recon_dist
	

def loss(inputs, z_mean, z_logvar, reconstruction_distribution):
	# Loss
	# Reconstruction error. (all log(p) are in [-\infty, 0]). 
	log_px_given_z = tf.reduce_mean(tf.reduce_sum(recon_dist.log_prob(inputs), axis = 1), name='reconstruction_error')
	# Regularization: Kulback-Leibler divergence between approximate posterior, q(z|x), and isotropic gauss prior p(z)=N(z,mu,sigma*I).
	KL_qp = tf.reduce_mean(kl_normal2_stdnormal(z_mean, z_logvar, eps=eps), name='kl_divergence')

	# Averaging over samples.  
	loss = tf.reduce_mean(log_px_given_z - KL_qp, name="lower_bound")
	
	tf.add_to_collection('losses', loss)
	tf.add_to_collection('losses', KL_qp)
	tf.add_to_collection('losses', log_px_given_z)

	return loss



def training(loss, learning_rate):
	"""Sets up the training Ops.

	Creates a summarizer to track the loss over time in TensorBoard.

	Creates an optimizer and applies the gradients to all trainable variables.

	The Op returned by this function is what must be passed to the
	`sess.run()` call to cause the model to train.

	Args:
	loss: Loss tensor, from loss().
	learning_rate: The learning rate to use for gradient descent.

	Returns:
	train_op: The Op for training.
	"""
	# Add a scalar summary for the snapshot loss.
	tf.summary.scalar('loss', loss)
	# Create the gradient descent optimizer with the given learning rate.
	# Make sure that the Updates of the moving_averages in batch_norm layers
	# are performed before the train_step.
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	if update_ops:
		updates = tf.group(*update_ops)
		with tf.control_dependencies([updates]):
			# Optimizer and training objective of negative loss
			optimizer = tf.train.AdamOptimizer(learning_rate)

			# Create a variable to track the global step.
			global_step = tf.Variable(0, name='global_step', trainable=False)
			
			# Use the optimizer to apply the gradients that minimize the loss
			# (and also increment the global step counter) as a single training step.
			train_op = optimizer.minimize(-loss, global_step=global_step)
	
	return train_op

def evaluation(loss):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  return loss


def train(train_data, valid_data, number_of_epochs=50, batch_size=100, learning_rate=1e-3):
	# Train
	M = train_data.number_of_examples 

	#train_losses, valid_losses = [], []
	feed_dict_train = {'x_in:0': train_data, 'phase:0': False}
	feed_dict_valid = {'x_in:0': valid_data, 'phase:0': False}
	for epoch in range(number_of_epochs):
		shuffled_indices = numpy.random.permutation(M)	
		for i in range(0, M, batch_size):
            subset = shuffled_indices[i:(i + batch_size)]
			batch = train_data[subset]
			_, batch_loss = sess.run([train_step, loss], feed_dict={'x_in:0': batch, 'phase:0': True})
		
		train_loss = sess.run(loss, feed_dict=feed_dict_train)
		valid_loss = sess.run(loss, feed_dict=feed_dict_valid)


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
		outputs = fully_connected(inputs, num_outputs=num_outputs, activation_fn=None, scope='DENSE')
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
