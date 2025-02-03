import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

def identity(x):
    return x

def threshold(x, threshold):
    return tf.where(x>threshold, tf.ones(tf.shape(x)), tf.zeros(tf.shape(x)))

def CAL_prediction(pi_tm1, k_pi_tm1):

    return tf.einsum("np,npm->nm", pi_tm1, k_pi_tm1)

def CAL_update(pi_t_tm1, g_t, y_t):

    mu_t = tf.einsum("np,npm->nm", pi_t_tm1, g_t)
    mu_t_in_y = tf.einsum("np,np->n", y_t, mu_t)

    emission_matrix_in_y = tf.einsum("np,nmp->nm", y_t, g_t)

    pi_t = tf.einsum("nm,nm->nm", pi_t_tm1, emission_matrix_in_y)
    pi_t = tf.einsum("nm,n->nm", pi_t, 1/tf.where(mu_t_in_y==0, tf.ones(tf.shape(mu_t_in_y)), mu_t_in_y))

    return pi_t, mu_t, tf.math.log(mu_t_in_y)

def CAL_without_extra(ibm, parameters, y):

	T = tf.shape(y)[0]

	pi_0 = ibm.pi_0(parameters)
	# nu_0 = ibm.pi_0(parameters)
	mu_0 = tf.zeros((tf.shape(pi_0)[0], ibm.M+1), dtype = tf.float32)
	log_likelihood = tf.zeros(tf.shape(pi_0)[0])

	def body(input, t):

		pi_tm1, _, _ = input

		k_pi_tm1 = ibm.K_x(parameters, pi_tm1)

		# nu_t     = CAL_prediction(nu_tm1,   k_pi_tm1)
		pi_t_tm1 = CAL_prediction(pi_tm1, k_pi_tm1)

		g_t = ibm.G_t(parameters)
		pi_t, mu_t, log_likelihood_increment = CAL_update(pi_t_tm1, g_t, y[t,...])

		return pi_t, mu_t, log_likelihood_increment

	Pi, Mu, Log_likelihood = tf.scan(body, tf.range(0, T), initializer = (pi_0, mu_0, log_likelihood))

	return tf.concat((tf.expand_dims(pi_0, axis = 0), Pi), axis = 0), Mu, tf.reduce_sum(tf.reduce_sum(Log_likelihood, axis = 1)) #tf.reduce_sum(tf.math.log(tf.reduce_mean(tf.math.exp(Log_likelihood), axis = 1)))

def CAL_with_extra(ibm, parameters, y):

	T = tf.shape(y)[0]

	pi_0 = ibm.pi_0(parameters)
	# nu_0 = ibm.pi_0(parameters)
	mu_0 = tf.zeros((tf.shape(pi_0)[0], ibm.M+1), dtype = tf.float32)
	log_likelihood = tf.zeros(tf.shape(pi_0)[0])
	extra_state = ibm.extra_state

	def body(input, t):

		pi_tm1, _, _, extra_state = input

		k_pi_tm1 = ibm.K_x(parameters, pi_tm1, extra_state)

		# nu_t     = CAL_prediction(nu_tm1,   k_pi_tm1)
		pi_t_tm1 = CAL_prediction(pi_tm1, k_pi_tm1)

		g_t = ibm.G_t(parameters, extra_state)
		pi_t, mu_t, log_likelihood_increment = CAL_update(pi_t_tm1, g_t, y[t,...])

		extra_state = ibm.extra(parameters, pi_t_tm1, y[t,...], extra_state)

		return pi_t, mu_t, log_likelihood_increment, extra_state

	Pi, Mu, Log_likelihood, _ = tf.scan(body, tf.range(0, T), initializer = (pi_0, mu_0, log_likelihood, extra_state))

	return tf.concat((tf.expand_dims(pi_0, axis = 0), Pi), axis = 0), Mu, tf.reduce_sum(tf.reduce_sum(Log_likelihood, axis = 1)) 

def CAL(ibm, parameters, y):

	if hasattr(ibm, 'extra') and callable(getattr(ibm, 'extra')):
		return CAL_with_extra(ibm, parameters, y)

	else:
		return CAL_without_extra(ibm, parameters, y)

@tf.function(jit_compile=True)
def CAL_compiled(ibm, parameters, y):

	if hasattr(ibm, 'extra') and callable(getattr(ibm, 'extra')):
		return CAL_with_extra(ibm, parameters, y)

	else:
		return CAL_without_extra(ibm, parameters, y)
	
@tf.function(jit_compile=True)
def CAL_loss_compiled(ibm, parameters, y):

	if hasattr(ibm, 'extra') and callable(getattr(ibm, 'extra')):
		_, _, log_likelihood = CAL_with_extra(ibm, parameters, y)

	else:
		_, _, log_likelihood = CAL_without_extra(ibm, parameters, y)
		
	return -log_likelihood/tf.cast(ibm.N, dtype = tf.float32)

# @tf.function(jit_compile=True)
# def CAL_joint_likelihood(ibm, parameters, X, Y):

# 	T = tf.shape(X)[0]
	
# 	pi_0 = ibm.pi_0(parameters)
# 	p_x_0 = tf.einsum("ni,ni->n", X[0,...], pi_0)

# 	def body(input, t):

# 		pi_tm1, _ = input

# 		x_tm1 = X[t-1,...]
# 		x_t   = X[t,...]
# 		y_t   = Y[t-1,...]

# 		k_pi_tm1 = ibm.K_x(parameters, pi_tm1)
# 		p_x_tm1     = tf.einsum("ni,nij->nj", x_tm1,   k_pi_tm1)
# 		p_x_tm1_x_t = tf.einsum("ni,ni->n",   p_x_tm1, x_t)

# 		pi_t_tm1 = CAL_prediction(pi_tm1, k_pi_tm1)

# 		g_t = ibm.G_t(parameters)
# 		g_x_tm1     = tf.einsum("ni,nij->nj", x_tm1,   g_t)
# 		g_x_tm1_x_t = tf.einsum("ni,ni->n",   g_x_tm1, y_t)

# 		pi_t, _, _ = CAL_update(pi_t_tm1, g_t, y_t)

# 		return pi_t, tf.math.log(p_x_tm1_x_t) + tf.math.log(g_x_tm1_x_t)

# 	_, p_X_Y = tf.scan(body, tf.range(1, T), initializer = (pi_0, tf.math.log(p_x_0)))

# 	return tf.concat((tf.expand_dims(tf.math.log(p_x_0), axis = 0), p_X_Y), axis = 0)

def CAL_loss(ibm, parameters, y):

    _, _, log_likelihood = CAL(ibm, parameters, y)

    return -log_likelihood/tf.cast(ibm.N, dtype = tf.float32)

@tf.function(jit_compile=True)
def CAL_grad_loss(ibm, parameters, y, learning_parameters):

    with tf.GradientTape() as g:
        # for key in learning_parameters:
        #     g.watch(parameters[key])

        loss = CAL_loss(ibm, parameters, y)

    return loss, g.gradient(loss, [parameters[key] for key in learning_parameters])

def CAL_inference(ibm, parameters, y, learning_parameters, optimizer, n_gradient_steps, seed_optim = None, initialization = "random"):

	parameters_list = {}
	loss_list = []

	if initialization == "random":

		seed_optim, seed_carry = tfp.random.split_seed( seed_optim, n = 2, salt = "seed_for_optimization")
		for key in learning_parameters.keys():

			parameters[key] = tf.Variable(tfp.distributions.Normal(loc = 0.0, scale = 0.5).sample(learning_parameters[key], seed = seed_optim), dtype = tf.float32)

			seed_optim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_optimization")

		current_loss = CAL_loss(ibm, parameters, y)

		while tf.math.is_nan(current_loss):
			for key in learning_parameters.keys():

				parameters[key] = tf.Variable(tfp.distributions.Normal(loc = 0.0, scale = 0.5).sample(learning_parameters[key], seed = seed_optim), dtype = tf.float32)

			seed_optim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_optimization")

			current_loss = CAL_loss(ibm, parameters, y)

	elif initialization == "parameters":

		for key in learning_parameters.keys():

			parameters[key] = tf.Variable(parameters[key])

		current_loss = CAL_loss(ibm, parameters, y)

	# else:
	# 	for key in learning_parameters:

	# 		parameters_list[key] = []
	# 		parameters_list[key].append(parameters[key])

	# 	current_loss = CAL_loss(ibm, parameters, y)

	loss_list.append(current_loss.numpy())

	for key in learning_parameters.keys():
		parameters_list[key] = []
		parameters_list[key].append(parameters[key].numpy())

	for i in range(n_gradient_steps):
		loss, grad = CAL_grad_loss(ibm, parameters, y, learning_parameters)
		optimizer.apply_gradients(zip(grad, [parameters[key] for key in learning_parameters]))

		# if tf.math.is_nan(loss):
		# 	break
		# else:
		# 	loss_list.append(loss.numpy())
		loss_list.append(loss.numpy())
            
		for key in learning_parameters: 
			parameters_list[key].append(parameters[key].numpy())

	loss_tensor = np.stack(loss_list)
	parameters_tensor = {}
	for key in learning_parameters: 
		parameters_tensor[key] = np.stack(parameters_list[key])

	return loss_tensor, parameters_tensor

# if __name__ == "__main__":

# 	import sys
# 	sys.path.append('CAL/Scripts/')
# 	from model import *

# 	locations  = tf.convert_to_tensor(np.load("CAL/Data/FM/locations_FM_cumbria.npy"), dtype = tf.float32)
# 	covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/covariates_FM_cumbria.npy"), dtype = tf.float32)

# 	batch_size = 5000

# 	parameters = {"prior_infection":tf.convert_to_tensor([1-0.001, 0.001, 0.0, 0.0], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(1.5)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(1.2)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(1.3)], dtype = tf.float32),
# 		"log_psi":tf.convert_to_tensor([np.log(1.5)], dtype = tf.float32),
# 		"log_q_rate":tf.convert_to_tensor([np.log(0.8)], dtype = tf.float32),
# 		"log_r_rate":tf.convert_to_tensor([np.log(0.5)], dtype = tf.float32),
# 		"logit_prob_testing":logit(
# 			tf.convert_to_tensor([0.0, 0.5, 1.0, 1.0], dtype = tf.float32)),}
	
# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria.npy"), dtype = tf.float32)[20:,...]

# 	FM_ibm = FM_SIQR(locations, covariates, batch_size)

# 	log_likelihood = CAL_compiled(FM_ibm, parameters, Y)

# 	print(log_likelihood)

# if __name__ == "__main__":

# 	import sys
# 	sys.path.append('CAL/Scripts/')
# 	from model import *

# 	locations  = tf.convert_to_tensor(np.load("CAL/Data/FM/locations_FM_cumbria.npy"), dtype = tf.float32)
# 	covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/covariates_FM_cumbria.npy"), dtype = tf.float32)

# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria.npy"), dtype = tf.float32)
# 	time_before_infection = tf.cast(tf.shape(Y)[0], tf.float32) - tf.reduce_sum(tf.math.cumsum( Y[...,3], axis = 0), axis = 0)

# 	batch_size = 5000

# 	parameters = {"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 	        "log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32),
# 		"log_gamma":tf.convert_to_tensor([np.log(0.25)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.0001)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}
	

# 	FM_ibm = FM_SINR(locations, covariates, batch_size)

# 	log_likelihood = CAL_compiled(FM_ibm, parameters, Y)

# 	n_gradient_steps = 200

# 	par_to_upd = {"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 		"log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32),
# 		"log_gamma":tf.convert_to_tensor([np.log(0.25)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.0001)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}

# 	learning_parameters = {"log_delta":1, "log_zeta":1, "log_xi":1, "log_chi":1, "log_psi":1, "log_gamma":1,"log_epsilon":1}
# 	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.5)

# 	loss_tensor, parameters_tensor = CAL_inference(FM_ibm, par_to_upd, Y[:100,...], learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

# 	print(log_likelihood)

# if __name__ == "__main__":

# 	import time

# 	import sys
# 	sys.path.append('CAL/Scripts/')
# 	from FM_model import *

# 	indexes = tf.convert_to_tensor(np.load("CAL/Data/FM/indexes_FM_cumbria.npy"), dtype = tf.int64)
# 	values  = tf.convert_to_tensor(np.load("CAL/Data/FM/values_FM_cumbria.npy"), dtype = tf.float32)
# 	covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/covariates_FM_cumbria.npy"), dtype = tf.float32)

# 	# Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria.npy"), dtype = tf.float32)[:10,...]
# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria_SIQR.npy"), dtype = tf.float32)
# 	time_before_infection = tf.cast(tf.shape(Y)[0], tf.float32) - tf.reduce_sum(tf.math.cumsum( Y[...,2], axis = 0), axis = 0)

# 	parameters = {"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 	        "log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.Variable(tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32)),
# 		"log_q_rate":tf.convert_to_tensor([np.log(0.25)], dtype = tf.float32),
# 		"log_r_rate":tf.convert_to_tensor([np.log(0.25)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.8, 1.0, 1.0], dtype = tf.float32)),}
	
# 	# FM_ibm = sparse_FM_SINR(values, indexes, covariates)
# 	FM_ibm = sparse_FM_SIQR(values, indexes, covariates)

# 	start = time.time()
# 	loss = CAL(FM_ibm, parameters, Y[:100,...])[2]
# 	print(time.time()-start)
# 	print(loss)

# 	start = time.time()
# 	loss = CAL_compiled(FM_ibm, parameters, Y[:100,...])[2]
# 	print(time.time()-start)
# 	print(loss)

# 	n_gradient_steps = 200

# 	par_to_upd = {"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 	        "log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.Variable(tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32)),
# 		"log_q_rate":tf.convert_to_tensor([np.log(0.25)], dtype = tf.float32),
# 		"log_r_rate":tf.convert_to_tensor([np.log(0.25)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.8, 1.0, 1.0], dtype = tf.float32)),}

# 	learning_parameters = { "log_delta":1, "log_zeta":1, "log_xi":1, "log_chi":1, "log_psi":1, 
# 				"log_q_rate":1, "log_r_rate":1, "log_epsilon":1}
# 	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# 	loss_tensor, parameters_tensor = CAL_inference(FM_ibm, par_to_upd, Y[:10,...], learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

# if __name__ == "__main__":
# 	import time

# 	N_pop = 1000
# 	covariates  = tf.convert_to_tensor(np.load("CAL/Data/GraphInference/Input/covariates.npy"), dtype = tf.float32)[:N_pop,:]
# 	communities = tf.convert_to_tensor(np.load("CAL/Data/GraphInference/Input/communities.npy"), dtype = tf.float32)[:N_pop,:]

# 	N = tf.shape(covariates)[0]

# 	parameters = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
# 		"beta_l":tf.convert_to_tensor([-1.0, +2.0], dtype = tf.float32),
# 		"beta_g":tf.convert_to_tensor([-1.0, -1.0], dtype = tf.float32),
# 		#       "logit_B":logit(B),
# 		"log_graph":tf.math.log(tf.convert_to_tensor([0.1], dtype = tf.float32)),
# 		"logit_sensitivity":logit(
# 			tf.convert_to_tensor([0.9], dtype = tf.float32)),
# 		"logit_specificity":logit(
# 			tf.convert_to_tensor([0.95], dtype = tf.float32)),
# 		"logit_prob_testing":logit(
# 			tf.convert_to_tensor([0.2, 0.5], dtype = tf.float32)),
# 		"log_epsilon":tf.math.log(
# 			tf.convert_to_tensor([0.001], dtype = tf.float32)),}

# 	SIS = sbm_SIS(communities, covariates)

# 	T    = 200
# 	start = time.time()
# 	X, Y = simulator(SIS, parameters, T)
# 	print(time.time()-start)

# 	X, Y = X[:,0,...], Y[:,0,...]

# 	Pi, Mu, log_likelihood = CAL(SIS, parameters, Y)
# 	print("hello")





