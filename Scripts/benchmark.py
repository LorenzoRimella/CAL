import tensorflow as tf
import numpy as np

class IBM_AR_common():

	def __init__(self, covariates):

		self.covariates = covariates

	def IBM_log_likelihood(self, parameters, Y):

		p_logistic_1t = tf.math.sigmoid(parameters["logit_lambda"])/(1+tf.exp(-parameters["gamma_Y"]*Y[:-1,...] - tf.einsum("c,nc->n", parameters["gamma_W"], self.covariates)))

		p_logistic_0  = tf.expand_dims( tf.math.sigmoid(parameters["logit_lambda"])/(1+tf.exp(-tf.einsum("c,nc->n", parameters["gamma_W"], self.covariates))), axis = 0)
		p_logistic    = tf.concat((p_logistic_0, p_logistic_1t), axis = 0)

		AR_likelihood_nt = Y*p_logistic + (1-Y)*(1-p_logistic)

		return tf.math.log(AR_likelihood_nt)

class IBM_AR_local():

	def __init__(self, covariates, communities):

		self.covariates  = covariates
		self.communities = communities

	def IBM_log_likelihood(self, parameters, Y):

		lambda_n  = tf.einsum("c...,nc->n...", tf.math.sigmoid(parameters["logit_lambda"]), self.communities)
		gamma_Y_n = tf.einsum("c...,nc->n...", parameters["gamma_Y"], self.communities)
		gamma_W_n = tf.einsum("c...,nc->n...", parameters["gamma_W"], self.communities)

		p_logistic_1t = lambda_n/(1+tf.exp(-gamma_Y_n*Y[:-1,...] - tf.einsum("nc,nc->n", gamma_W_n, self.covariates)))

		p_logistic_0  = tf.expand_dims( lambda_n/(1+tf.exp(-tf.einsum("nc,nc->n", gamma_W_n, self.covariates))), axis = 0)
		p_logistic    = tf.concat((p_logistic_0, p_logistic_1t), axis = 0)

		AR_likelihood_nt = Y*p_logistic + (1-Y)*(1-p_logistic)

		return tf.math.log(AR_likelihood_nt)
	
def logistic_log_likelihood(IBM_AR, parameters, Y):

	return tf.reduce_sum(IBM_AR.IBM_log_likelihood(parameters, Y))


def logistic_loss(IBM_AR, parameters, Y):

	T = tf.cast(tf.shape(Y)[0], dtype = tf.float32)

	return -T*tf.reduce_mean(IBM_AR.IBM_log_likelihood(parameters, Y))

@tf.function(jit_compile=True)
def grad_logistic_loss(IBM_AR, parameters, Y):

	with tf.GradientTape() as g:

		loss = logistic_loss(IBM_AR, parameters, Y)

	return loss, g.gradient(loss, [parameters[key] for key in parameters.keys()])

def IBM_AR_inference(IBM_AR, parameters_to_learn, Y, n_gradient_steps = 10000, optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)):

	loss_history = []
	parameters_to_learn_history = {}

	for key in parameters_to_learn.keys():

		parameters_to_learn_history[key] = []
		parameters_to_learn_history[key].append(parameters_to_learn[key].numpy())

		parameters_to_learn[key] = tf.Variable(parameters_to_learn[key])
	
	for i in range(n_gradient_steps):
		loss, grad = grad_logistic_loss(IBM_AR, parameters_to_learn, Y)

		optimizer.apply_gradients(zip(grad, [parameters_to_learn[key] for key in parameters_to_learn.keys()]))

		loss_history.append(loss.numpy())

		for key in parameters_to_learn.keys():

			parameters_to_learn_history[key].append(parameters_to_learn[key].numpy())

	return np.stack(loss_history), {key:np.stack(parameters_to_learn_history[key]) for key in parameters_to_learn_history.keys()}, parameters_to_learn
