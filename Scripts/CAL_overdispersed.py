import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from CAL import CAL_prediction, CAL_update

@tf.function(jit_compile=True)
def ovd_CAL_prediction(pi_tm1, k_pi_tm1):

    return tf.einsum("...np,...npm->...nm", pi_tm1, k_pi_tm1)

@tf.function(jit_compile=True)
def ovd_CAL_update(pi_t_tm1, g_t, y_t):

    mu_t = tf.einsum("...np,...npm->...nm", pi_t_tm1, g_t)
    mu_t_in_y = tf.einsum("...np,...np->...n", y_t, mu_t)

    emission_matrix_in_y = tf.einsum("...np,...nmp->...nm", y_t, g_t)

    pi_t = tf.einsum("...nm,...nm->...nm", pi_t_tm1, emission_matrix_in_y)
    pi_t = tf.einsum("...nm,...n->...nm", pi_t, 1/tf.where(mu_t_in_y==0, tf.ones(tf.shape(mu_t_in_y)), mu_t_in_y))

    return pi_t, mu_t, tf.math.log(mu_t_in_y)

def ovd_CALSMC(ovd_ibm, parameters, y, blocking_function, unblocking_function, n_particles = 256, seed_smc = None):
	
	T = tf.shape(y)[0]

	parameters_SMC = parameters.copy()

	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

	for key in ovd_ibm.ovd_prior_shape.keys():
		parameters_SMC[key] = ovd_ibm.ovd_prior[key].sample((n_particles, ovd_ibm.ovd_prior_shape[key], T), seed = seed_smc_run[0])

	pi_0 = ovd_ibm.pi_0(parameters_SMC)
	pi_0 = tf.ones((n_particles, 1, 1))*tf.expand_dims(pi_0, axis = 0)

	recursive_params = []

	for key in ovd_ibm.ovd_prior_shape.keys():
		recursive_params.append(parameters_SMC[key][:,:,0])

	recursive_params = tf.concat(recursive_params, axis = -1)
	
	log_likelihood = tf.zeros(1)
	ESS = tf.zeros(tf.shape(blocking_function(ovd_ibm, tf.ones(tf.shape(y)[1]))))

	def body(input, t):

		_, pi_tm1, _, _ = input

		k_pi_tm1 = ovd_ibm.K_x(parameters_SMC, pi_tm1, t)

		pi_t_tm1 = ovd_CAL_prediction(pi_tm1, k_pi_tm1)

		g_t = ovd_ibm.G_t(parameters_SMC, t)
		g_t = tf.ones((n_particles, 1, 1, 1))*tf.expand_dims(g_t, axis = 0)

		pi_t, _, log_likelihood_increment = ovd_CAL_update(pi_t_tm1, g_t, y[t-1,...])

		log_likelihood_increment = tf.where(log_likelihood_increment<-5000, -5000, log_likelihood_increment)
		log_likelihood_block = blocking_function(ovd_ibm, log_likelihood_increment)

		log_weights_pn = tf.transpose(log_likelihood_block)
		shifted_weights_pn = tf.exp(log_weights_pn-tf.reduce_max(log_weights_pn, axis = 1, keepdims = True))
		norm_weights_pn = shifted_weights_pn/tf.reduce_sum(shifted_weights_pn, axis = 1, keepdims=True)
		ESS = 1/tf.reduce_sum(tf.math.pow(norm_weights_pn, 2), axis =1)

		# tf.reduce_sum(tf.math.log(tf.reduce_mean(weights_np, axis =0)))
		log_likelihood_SMC = tf.reduce_sum(tf.math.log(tf.reduce_mean(shifted_weights_pn, axis =1)) + tf.reduce_max(log_weights_pn, axis = 1))

		indeces = tfp.distributions.Categorical(probs = norm_weights_pn).sample(n_particles, seed = seed_smc_run[t])
		
		recursive_params_t = [] 
		for key in ovd_ibm.ovd_prior_shape.keys():
			resample_param = tf.transpose(tf.gather(tf.transpose(parameters_SMC[key][...,t-1], [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
			# mask = tf.expand_dims(tf.expand_dims(tf.one_hot(t, T), axis = 0), axis = 0)
			# parameters_SMC[key] = parameters_SMC[key]*(1-mask) + tf.expand_dims(resample_param, axis = -1)*mask
			recursive_params_t.append(resample_param)

		recursive_params_t = tf.concat(recursive_params_t, axis = -1)

		indeces_n = unblocking_function(ovd_ibm, indeces)
		res_pi_t = tf.transpose(tf.gather(tf.transpose(pi_t, [1, 0, 2]), tf.transpose(indeces_n), axis = 1, batch_dims=1 ), [1, 0, 2])

		return recursive_params_t, res_pi_t, tf.expand_dims(log_likelihood_SMC, axis = 0), ESS

	parameters_recursive, Pi, Log_likelihood, ESS = tf.scan(body, tf.range(1, T+1), initializer = (recursive_params, pi_0, log_likelihood, ESS))

	Log_likelihood = tf.reduce_sum(tf.reduce_sum(Log_likelihood, axis = 1))

	return parameters_recursive, Pi, Log_likelihood, ESS


def ovd_CALSMC_likelihood(ovd_ibm, parameters, y, blocking_function, unblocking_function, n_particles = 256, seed_smc = None):
	
	T = tf.shape(y)[0]

	parameters_SMC = parameters.copy()

	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

	for key in ovd_ibm.ovd_prior_shape.keys():
		parameters_SMC[key] = ovd_ibm.ovd_prior[key].sample((n_particles, ovd_ibm.ovd_prior_shape[key], T), seed = seed_smc_run[0])

	pi_0 = ovd_ibm.pi_0(parameters_SMC)
	pi_0 = tf.ones((n_particles, 1, 1))*tf.expand_dims(pi_0, axis = 0)

	recursive_params = []

	for key in ovd_ibm.ovd_prior_shape.keys():
		recursive_params.append(parameters_SMC[key][:,:,0])

	recursive_params = tf.expand_dims(tf.concat(recursive_params, axis = -1), axis = -1)
	
	log_likelihood = tf.zeros(1)
	ESS = n_particles*tf.expand_dims(tf.ones(tf.shape(blocking_function(ovd_ibm, tf.ones(tf.shape(y)[1])))), axis = -1)


	def cond(input, t):

		return t<T+1

	def body(input, t):

		# print(t)

		recursive_params_tm1, pi_tm1, log_likelihood, ESS_tm1 = input

		k_pi_tm1 = ovd_ibm.K_x(parameters_SMC, pi_tm1, t)

		pi_t_tm1 = ovd_CAL_prediction(pi_tm1, k_pi_tm1)

		g_t = ovd_ibm.G_t(parameters_SMC, t)
		g_t = tf.ones((n_particles, 1, 1, 1))*tf.expand_dims(g_t, axis = 0)

		pi_t, _, log_likelihood_increment = ovd_CAL_update(pi_t_tm1, g_t, y[t-1,...])

		log_likelihood_increment = tf.where(log_likelihood_increment<-5000, -5000, log_likelihood_increment)
		log_likelihood_block = blocking_function(ovd_ibm, log_likelihood_increment)

		log_weights_pn = tf.transpose(log_likelihood_block)
		shifted_weights_pn = tf.exp(log_weights_pn-tf.reduce_max(log_weights_pn, axis = 1, keepdims = True))
		norm_weights_pn = shifted_weights_pn/tf.reduce_sum(shifted_weights_pn, axis = 1, keepdims=True)
		ESS_t = 1/tf.reduce_sum(tf.math.pow(norm_weights_pn, 2), axis =1)

		# tf.reduce_sum(tf.math.log(tf.reduce_mean(weights_np, axis =0)))
		log_likelihood_SMC = tf.reduce_sum(tf.math.log(tf.reduce_mean(shifted_weights_pn, axis =1)) + tf.reduce_max(log_weights_pn, axis = 1))

		indeces = tfp.distributions.Categorical(probs = norm_weights_pn).sample(n_particles, seed = seed_smc_run[t])
		
		recursive_params_t = [] 
		for key in ovd_ibm.ovd_prior_shape.keys():
			resample_param = tf.transpose(tf.gather(tf.transpose(parameters_SMC[key][...,t-1], [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
			# mask = tf.expand_dims(tf.expand_dims(tf.one_hot(t, T), axis = 0), axis = 0)
			# parameters_SMC[key] = parameters_SMC[key]*(1-mask) + tf.expand_dims(resample_param, axis = -1)*mask
			recursive_params_t.append(resample_param)

		recursive_params_t = tf.concat(recursive_params_t, axis = -1)

		indeces_n = unblocking_function(ovd_ibm, indeces)
		res_pi_t = tf.transpose(tf.gather(tf.transpose(pi_t, [1, 0, 2]), tf.transpose(indeces_n), axis = 1, batch_dims=1 ), [1, 0, 2])

		return (tf.concat((recursive_params_tm1, tf.expand_dims(recursive_params_t, axis = -1)), axis = -1), res_pi_t, log_likelihood + log_likelihood_SMC, tf.concat((ESS_tm1, tf.expand_dims(ESS_t, axis = -1)), axis = -1)), t+1

	output, _ = tf.while_loop(cond, body, loop_vars = ((recursive_params, pi_0, log_likelihood, ESS), 1))

	return output[0], output[-2], output[-1]


# def batched_indexing(Pi, indices, batch_size):
    
#     N = tf.shape(norm_weights)[0]
#     M = tf.shape(X_p)[-1]
#     P = tf.shape(X_p)[0]

#     batch_number = tf.cast(tf.math.floor(N/batch_size), tf.int32)

#     def body(input, i):

#         sub_P = tf.cast(P/4, tf.int32)
        
#         indeces_1 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[0])
#         indeces_2 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[1])
#         indeces_3 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[2])
#         indeces_4 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[3])
        
#         indeces = tf.concat((indeces_1, indeces_2, indeces_3, indeces_4), axis = 0)

#         return tf.transpose(tf.gather(tf.transpose(X_p[:,(i*batch_size):((i*batch_size) + batch_size),:], [1, 0, 2]), tf.transpose(indeces), axis = 1, batch_dims = 1 ), [1, 0, 2])
    
#     output = tf.scan(body, tf.range(0, batch_number, dtype=tf.int64), initializer = tf.zeros((P, batch_size, M)))
#     output_transposed = tf.transpose(output, perm=[1, 0, 2, 3])

#     output_reshaped = tf.reshape(output_transposed, [P, N, M])

#     return output_reshaped

# def ovd_CALSMC_batched(ovd_ibm, parameters, y, blocking_function, unblocking_function, n_particles = 256, batch_size=10000, seed_smc = None):
	
# 	T = tf.shape(y)[0]

# 	parameters_SMC = parameters.copy()

# 	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

# 	for key in ovd_ibm.ovd_prior_shape.keys():
# 		parameters_SMC[key] = ovd_ibm.ovd_prior[key].sample((n_particles, ovd_ibm.ovd_prior_shape[key], T), seed = seed_smc_run[0])

# 	pi_0 = ovd_ibm.pi_0(parameters_SMC)
# 	pi_0 = tf.ones((n_particles, 1, 1))*tf.expand_dims(pi_0, axis = 0)

# 	recursive_params = []

# 	for key in ovd_ibm.ovd_prior_shape.keys():
# 		recursive_params.append(parameters_SMC[key][:,:,0])

# 	recursive_params = tf.concat(recursive_params, axis = -1)
	
# 	log_likelihood = tf.zeros(1)

# 	def body(input, t):

# 		_, pi_tm1, _ = input

# 		k_pi_tm1 = ovd_ibm.K_x(parameters_SMC, pi_tm1, t)

# 		pi_t_tm1 = ovd_CAL_prediction(pi_tm1, k_pi_tm1)

# 		g_t = ovd_ibm.G_t(parameters_SMC, t)
# 		g_t = tf.ones((n_particles, 1, 1, 1))*tf.expand_dims(g_t, axis = 0)

# 		pi_t, _, log_likelihood_increment = ovd_CAL_update(pi_t_tm1, g_t, y[t-1,...])

# 		log_likelihood_increment = tf.where(log_likelihood_increment<-5000, -5000, log_likelihood_increment)
# 		log_likelihood_block = blocking_function(ovd_ibm, log_likelihood_increment)

# 		log_weights_pn = tf.transpose(log_likelihood_block)
# 		shifted_weights_pn = tf.exp(log_weights_pn-tf.reduce_max(log_weights_pn, axis = 1, keepdims = True))
# 		norm_weights_pn = shifted_weights_pn/tf.reduce_sum(shifted_weights_pn, axis = 1, keepdims=True)

# 		# tf.reduce_sum(tf.math.log(tf.reduce_mean(weights_np, axis =0)))
# 		log_likelihood_SMC = tf.reduce_sum(tf.math.log(tf.reduce_mean(shifted_weights_pn, axis =1)) + tf.reduce_max(log_weights_pn, axis = 1))

# 		indeces = tfp.distributions.Categorical(probs = norm_weights_pn).sample(n_particles, seed = seed_smc_run[t])
		
# 		recursive_params_t = [] 
# 		for key in ovd_ibm.ovd_prior_shape.keys():
# 			resample_param = tf.transpose(tf.gather(tf.transpose(parameters_SMC[key][...,t-1], [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
# 			# mask = tf.expand_dims(tf.expand_dims(tf.one_hot(t, T), axis = 0), axis = 0)
# 			# parameters_SMC[key] = parameters_SMC[key]*(1-mask) + tf.expand_dims(resample_param, axis = -1)*mask
# 			recursive_params_t.append(resample_param)

# 		recursive_params_t = tf.concat(recursive_params_t, axis = -1)

# 		indeces_n = unblocking_function(ovd_ibm, indeces)
# 		# res_pi_t = tf.transpose(tf.gather(tf.transpose(pi_t, [1, 0, 2]), tf.transpose(indeces_n), axis = 1, batch_dims=1 ), [1, 0, 2])

# 		res_pi_t = batched_indexing(pi_t, indeces_n, batch_size)

# 		return recursive_params_t, res_pi_t, tf.expand_dims(log_likelihood_SMC, axis = 0)

# 	parameters_recursive, _, Log_likelihood = tf.scan(body, tf.range(1, T+1), initializer = (recursive_params, pi_0, log_likelihood))

# 	Log_likelihood = tf.reduce_sum(tf.reduce_sum(Log_likelihood, axis = 1))

# 	return parameters_recursive, Log_likelihood

def ovd_CAL(ovd_ibm, parameters, y):

	T = tf.shape(y)[0]

	pi_0 = ovd_ibm.pi_0(parameters)
	# nu_0 = ibm.pi_0(parameters)
	mu_0 = tf.zeros((tf.shape(pi_0)[0], ovd_ibm.M+1), dtype = tf.float32)
	log_likelihood = tf.zeros(tf.shape(pi_0)[0])

	def body(input, t):

		pi_tm1, _, _ = input

		k_pi_tm1 = ovd_ibm.K_x(parameters, pi_tm1, t)

		# nu_t     = CAL_prediction(nu_tm1,   k_pi_tm1)
		pi_t_tm1 = CAL_prediction(pi_tm1, k_pi_tm1)

		g_t = ovd_ibm.G_t(parameters, t)
		pi_t, mu_t, log_likelihood_increment = CAL_update(pi_t_tm1, g_t, y[t-1,...])

		return pi_t, mu_t, log_likelihood_increment

	Pi, Mu, Log_likelihood = tf.scan(body, tf.range(1, T+1), initializer = (pi_0, mu_0, log_likelihood))

	Log_likelihood = tf.reduce_sum(tf.reduce_sum(Log_likelihood, axis = 1))

	return tf.concat((tf.expand_dims(pi_0, axis = 0), Pi), axis = 0), Mu, Log_likelihood

def ovd_CAL_prior(ovd_ibm, parameters):

	log_prior = tf.zeros(1)
	for key in ovd_SIR.ovd_prior_shape.keys():
		log_prior = log_prior + tf.reduce_sum(ovd_ibm.ovd_prior[key].log_prob(parameters[key]))
	
	return log_prior

def ovd_CAL_loss(ovd_ibm, parameters, y):

	T = tf.cast(tf.shape(y)[0], dtype = tf.float32)
	N = tf.cast(ovd_ibm.N, dtype = tf.float32)

	_, _, log_likelihood = ovd_CAL(ovd_ibm, parameters, y)
	
	return -log_likelihood/N #- ovd_CAL_prior(ovd_ibm, parameters)

@tf.function(jit_compile=True)
def ovd_CAL_grad_loss(ibm, parameters, y, learning_parameters):

    with tf.GradientTape() as g:

        loss = ovd_CAL_loss(ibm, parameters, y)

    return loss, g.gradient(loss, [parameters[key] for key in learning_parameters])

def ovd_CAL_inference(ibm, parameters, y, learning_parameters, optimizer, n_gradient_steps, seed_optim = None, initialization = "random"):

	parameters_list = {}
	loss_list = []

	if initialization == "random":

		seed_optim, seed_carry = tfp.random.split_seed( seed_optim, n = 2, salt = "seed_for_optimization")
		for key in learning_parameters.keys():

			parameters[key] = tf.Variable(tfp.distributions.Normal(loc = 0.0, scale = 0.5).sample(learning_parameters[key], seed = seed_optim), dtype = tf.float32)

			seed_optim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_optimization")

		current_loss = ovd_CAL_loss(ibm, parameters, y)

		while tf.math.is_nan(current_loss):
			for key in learning_parameters.keys():

				parameters[key] = tf.Variable(tfp.distributions.Normal(loc = 0.0, scale = 0.5).sample(learning_parameters[key], seed = seed_optim), dtype = tf.float32)

			seed_optim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_optimization")

			current_loss = ovd_CAL_loss(ibm, parameters, y)

	elif initialization == "parameters":

		for key in learning_parameters.keys():

			parameters[key] = tf.Variable(parameters[key])

		current_loss = ovd_CAL_loss(ibm, parameters, y)

	loss_list.append(current_loss.numpy())

	for key in learning_parameters.keys():
		parameters_list[key] = []
		parameters_list[key].append(parameters[key].numpy())

	for i in range(n_gradient_steps):
		loss, grad = ovd_CAL_grad_loss(ibm, parameters, y, learning_parameters)
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

# 	import time

# 	from model_overdispersed import *
# 	from model import *
# 	from CAL import *

# 	from scipy.spatial.distance import pdist, squareform

# 	import matplotlib.pyplot as plt

# 	local_autorities_covariates  = tf.convert_to_tensor(np.load("CAL/Data/FM/local_autorities_covariates.npy"), dtype = tf.float32)
# 	farms_covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/farms_covariates.npy"), dtype = tf.float32)
# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/cut_Y_FM.npy"), dtype = tf.float32)

# 	x_bound = [300, 400]
# 	y_bound = [300, 500]

# 	x_cond = tf.stack((local_autorities_covariates[:,0]>x_bound[0], local_autorities_covariates[:,0]<x_bound[1]), axis = -1)
# 	y_cond = tf.stack((local_autorities_covariates[:,1]>y_bound[0], local_autorities_covariates[:,1]<y_bound[1]), axis = -1)

# 	indexes_c  = tf.where(tf.reduce_all(tf.concat((x_cond, y_cond), axis = -1), axis = -1))[:,0]
# 	indexes_n = tf.where(tf.convert_to_tensor(np.isin(farms_covariates[:,-1], indexes_c)))[:,0]

# 	local_autorities_covariates = tf.gather(local_autorities_covariates, indexes_c, axis = 0)
# 	farms_covariates = tf.gather(farms_covariates, indexes_n, axis = 0)
# 	Y = tf.gather(Y, indexes_n, axis = 1)[:100,:]

# 	# plt.plot(tf.reduce_sum(Y, axis = 1)[:100,2])
# 	# plt.show()

# 	communities = farms_covariates[:,-1:]
# 	farms_covariates = farms_covariates[:,:2]

# 	current_communities = np.unique(communities)
# 	counter = 0
# 	for i in current_communities:

# 		communities = tf.where(communities==i, counter, communities)

# 		counter = counter + 1

# 	parameters = {
# 		"log_tau": tf.convert_to_tensor([-8], dtype = tf.float32),
# 		"log_beta": tf.convert_to_tensor([5], dtype = tf.float32),
# 		"b_S": tf.convert_to_tensor([0.35, 0.12], dtype = tf.float32),
# 		"b_I": tf.convert_to_tensor([0.25, 0.10], dtype = tf.float32),
# 		"log_phi": tf.convert_to_tensor([3], dtype = tf.float32),
# 		"log_gamma": tf.convert_to_tensor([2.5], dtype = tf.float32),
# 		"log_rho": tf.convert_to_tensor([14], dtype = tf.float32),
# 		"log_psi": tf.convert_to_tensor([-2.8], dtype = tf.float32),
# 		"log_epsilon": tf.convert_to_tensor([-13], dtype = tf.float32),
# 		"logit_prob_I_testing":tf.convert_to_tensor([0.74], dtype = tf.float32)}

# 	ovd_prior = {}
# 	ovd_prior["log_Xi"] = tfp.distributions.Normal(loc = 0.0, scale = 0.25)
	
# 	ovdSIR = ovd_FM_SIR_communities(local_autorities_covariates, communities, farms_covariates, ovd_prior)

# 	def blocking_function(ovd_ibm, log_likelihood_increment):

# 		return tf.einsum("...n,nc->...c", log_likelihood_increment, ovd_ibm.communities)
	
# 	def unblocking_function(ovd_ibm, indeces):

# 		return tf.einsum("...c,nc->...n", indeces, tf.cast(ovd_ibm.communities, tf.int32))

# 	parameters_SMC, log_likelihood, ESS = ovd_CALSMC(ovdSIR, parameters, Y, blocking_function, unblocking_function, 128)


# if __name__ == "__main__":
# 	import time

# 	from model_overdispersed import *
# 	from model import *
# 	from CAL import *

# 	from scipy.spatial.distance import pdist, squareform

# 	import matplotlib.pyplot as plt

# 	# local_autorities_covariates  = tf.convert_to_tensor(np.load("CAL/Data/FM/local_autorities_covariates.npy"), dtype = tf.float32)

# 	# x_bound = [300, 400]
# 	# y_bound = [300, 500]

# 	# x_cond = tf.stack((local_autorities_covariates[:,0]>x_bound[0], local_autorities_covariates[:,0]<x_bound[1]), axis = -1)
# 	# y_cond = tf.stack((local_autorities_covariates[:,1]>y_bound[0], local_autorities_covariates[:,1]<y_bound[1]), axis = -1)

# 	# indexes_c  = tf.where(tf.reduce_all(tf.concat((x_cond, y_cond), axis = -1), axis = -1))[:,0]
	

# 	# farms_covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/farms_covariates.npy"), dtype = tf.float32)
# 	# Y = tf.convert_to_tensor(np.load("CAL/Data/FM/cut_Y_FM.npy"), dtype = tf.float32)

# 	# indexes_n = tf.where(np.isin(farms_covariates[:,-1], indexes_c))[:,0]

# 	# local_autorities_covariates = tf.gather(local_autorities_covariates, indexes_c, axis = 0)
# 	# farms_covariates = tf.gather(farms_covariates, indexes_n, axis = 0)
# 	# Y = tf.gather(Y, indexes_n, axis = 1)

# 	# N = tf.shape(Y)[1]
# 	# communities = farms_covariates[:,-1:]
# 	# farms_covariates = farms_covariates[:,:2]

# 	local_autorities_covariates  = tf.convert_to_tensor(np.load("CAL/Data/FM/local_autorities_covariates.npy"), dtype = tf.float32)
# 	farms_covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/farms_covariates.npy"), dtype = tf.float32)
# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/cut_Y_FM.npy"), dtype = tf.float32)

# 	x_bound = [300, 400]
# 	y_bound = [300, 500]

# 	x_cond = tf.stack((local_autorities_covariates[:,0]>x_bound[0], local_autorities_covariates[:,0]<x_bound[1]), axis = -1)
# 	y_cond = tf.stack((local_autorities_covariates[:,1]>y_bound[0], local_autorities_covariates[:,1]<y_bound[1]), axis = -1)

# 	indexes_c  = tf.where(tf.reduce_all(tf.concat((x_cond, y_cond), axis = -1), axis = -1))[:,0]
# 	indexes_n = tf.where(tf.convert_to_tensor(np.isin(farms_covariates[:,-1], indexes_c)))[:,0]

# 	local_autorities_covariates = tf.gather(local_autorities_covariates, indexes_c, axis = 0)
# 	farms_covariates = tf.gather(farms_covariates, indexes_n, axis = 0)
# 	Y = tf.gather(Y, indexes_n, axis = 1)[:100,:]

# 	plt.plot(tf.reduce_sum(Y, axis = 1)[:100,2])
# 	plt.show()

# 	communities = farms_covariates[:,-1:]
# 	farms_covariates = farms_covariates[:,:2]

# 	parameters = {
# 		"log_tau":tf.math.log(
# 			tf.convert_to_tensor([np.exp(-9)], dtype = tf.float32)),
# 		"log_beta":tf.math.log(
# 			tf.convert_to_tensor([np.exp(4)], dtype = tf.float32)),
# 		"b_S":tf.convert_to_tensor([+0.35, 0.25], dtype = tf.float32),
# 		"b_I":tf.convert_to_tensor([+0.35, 0.25], dtype = tf.float32),
# 		"log_phi":tf.math.log(
# 			tf.convert_to_tensor([np.exp(2)], dtype = tf.float32)),
# 		"log_gamma":tf.math.log(
# 			tf.convert_to_tensor([np.exp(1)], dtype = tf.float32)),
# 		"log_rho":tf.math.log(
# 			tf.convert_to_tensor([np.exp(4)], dtype = tf.float32)),
# 		"log_psi":tf.math.log(
# 			tf.convert_to_tensor([np.exp(2)], dtype = tf.float32)),
# 		"log_epsilon":tf.math.log(
# 			tf.convert_to_tensor([np.exp(-10)], dtype = tf.float32)),
# 		"logit_prob_I_testing":logit(
# 			tf.convert_to_tensor([tf.math.sigmoid(0.69)], dtype = tf.float32))}
	
# 	SIR = FM_SIR(local_autorities_covariates, communities, farms_covariates)
# 	Pi, Mu, Log_likelihood = CAL(SIR, parameters, Y)

# 	ovd_prior = {"log_Xi": tfp.distributions.Normal(loc = 0.0, scale = 0.5)}
# 	ovd_SIR = ovd_FM_SIR(local_autorities_covariates, communities, farms_covariates, ovd_prior)

# 	# for key in ovd_SIR.ovd_prior_shape.keys():
# 	# 	parameters[key] = ovd_SIR.ovd_prior[key].sample((1, T))

# 	# ovd_X_noisy, ovd_Y_noisy = ovd_simulator(ovd_SIR, parameters, T, 1, 567)
# 	# plt.plot(tf.reduce_sum(ovd_X_noisy, axis = 2)[1:,0,1], color = "purple")

# 	# plt.plot(tf.reduce_sum(ovd_Y_noisy, axis = 2)[:,0,2], color = "purple")
# 	# plt.show()

# 	def blocking_function(ovd_ibm, log_likelihood_increment):

# 		return tf.reduce_sum(log_likelihood_increment, axis = -1, keepdims = True)
	
# 	def unblocking_function(ovd_ibm, indeces):

# 		return tf.einsum("...c,cn->...n", indeces, tf.ones((1, ovd_ibm.N), tf.int32))
	

# 	parameters_SMC, log_likelihood_test = ovd_CALSMC_likelihood(ovd_SIR, parameters, Y, blocking_function, unblocking_function, 128)

# 	# parameters_SMC, Pi, log_likelihood = ovd_CALSMC(ovd_SIR, parameters, Y, blocking_function, unblocking_function, 128)

# 	print("ciao")

# 	SMC_quantiles = np.quantile(parameters_SMC, (0.025, 0.975), axis = 0)

# 	plt.fill_between(np.linspace(1, T, T), SMC_quantiles[0,:], SMC_quantiles[1,:])
# 	plt.plot(np.linspace(1, T, T), parameters["log_Xi"][:], color = "red")
# 	plt.show()

# 	community = 0
# 	Pi_community = tf.einsum("...nm,nc->...cm", Pi, ovd_SIR.communities)
# 	X_community  = tf.einsum("...nm,nc->...cm", ovd_X_noisy[:,0,...], ovd_SIR.communities)

# 	Pi_quantiles = np.quantile(Pi_community[...,community,:], (0.025, 0.975), axis = 1)
# 	X_community_noisy = X_community[1:,community,:]
# 	plt.plot(np.linspace(1, T, T), X_community_noisy[...,1], color = "purple")
# 	plt.fill_between(np.linspace(1, T, T), Pi_quantiles[0,...,1], Pi_quantiles[1,...,1], color = "grey")
# 	print("END")

# if __name__ == "__main__":
# 	import time

# 	from model_overdispersed import *
# 	from scipy.spatial.distance import pdist, squareform

# 	import matplotlib.pyplot as plt

# 	N_pop = 10000

# 	covariates_sample = tf.convert_to_tensor(np.load("CAL/Data/SpatialInference/Input/covariates.npy"), dtype = tf.float32)[:N_pop,...]

# 	parameters = {"prior_infection":tf.convert_to_tensor([1-0.05, 0.05], dtype = tf.float32),
# 		"log_beta":tf.math.log(
# 			tf.convert_to_tensor([0.5], dtype = tf.float32)),
# 		"b_S":tf.convert_to_tensor([+0.5], dtype = tf.float32),
# 		"b_I":tf.convert_to_tensor([-0.5], dtype = tf.float32),
# 		"b_R":tf.convert_to_tensor([+1.0], dtype = tf.float32),
# 		"log_gamma":tf.math.log(
# 			tf.convert_to_tensor([0.5], dtype = tf.float32)),
# 		"logit_prob_testing":logit(
# 			tf.convert_to_tensor([0.5, 0.8], dtype = tf.float32)),
# 		"logit_specificity":logit( tf.convert_to_tensor([0.95], dtype = tf.float32)),
# 		"logit_sensitivity":logit( tf.convert_to_tensor([0.95], dtype = tf.float32)),
# 			}
	
# 	T = 100

# 	ovd_prior = {"log_Xi": tfp.distributions.Normal(loc = 0.0, scale = 0.5)}
	
# 	ovd_SIS = ovd_logistic_SIS(covariates_sample, ovd_prior)

# 	for key in ovd_SIS.ovd_prior_shape.keys():
# 		parameters[key] = ovd_SIS.ovd_prior[key].sample((ovd_SIS.ovd_prior_shape[key], T))

# 	ovd_X_noisy, ovd_Y_noisy = ovd_simulator(ovd_SIS, parameters, T)
# 	plt.plot(tf.reduce_sum(ovd_X_noisy, axis = 2)[:,0,1], color = "purple")
# 	plt.show()

# 	plt.plot(tf.reduce_sum(ovd_Y_noisy, axis = 2)[:,0,2], color = "purple")
# 	plt.show()

# 	def blocking_function(ovd_ibm, log_likelihood_increment):

# 		return tf.reduce_sum(log_likelihood_increment, axis = -1, keepdims=True)
	
# 	def unblocking_function(ovd_ibm, indeces):

# 		return tf.einsum("...c,nc->...n", indeces, tf.ones((ovd_ibm.N, 1), tf.int32))

# 	parameters_SMC, log_likelihood = ovd_CALSMC(ovd_SIS, parameters, ovd_Y_noisy[:,0,...], blocking_function, unblocking_function, 100)

# 	SMC_quantiles = np.quantile(parameters_SMC, (0.025, 0.975), axis = 1)

# 	community = 0
# 	plt.fill_between(np.linspace(1, T, T), SMC_quantiles[0,:,community], SMC_quantiles[1,:,community])
# 	plt.plot(np.linspace(1, T, T), parameters["log_Xi"][community,:], color = "red")
# 	plt.show()

# 	learning_parameters = {"log_Xi":(ovd_SIS.ovd_prior_shape["log_Xi"], T)} 

# 	n_gradient_steps = 2000

# 	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# 	par_to_upd = parameters.copy()
# 	par_to_upd["log_Xi"] = ovd_SIS.ovd_prior["log_Xi"].sample((ovd_SIS.ovd_prior_shape["log_Xi"], T))

# 	loss_tensor, parameters_tensor = ovd_CAL_inference(ovd_SIS, par_to_upd, ovd_Y_noisy[:,0,...], learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

# 	plt.plot(loss_tensor)
# 	plt.axhline(y = ovd_CAL_loss(ovd_SIR, parameters, ovd_Y_noisy[:,0,...]), color = "red")

# 	community = 0
# 	for iteration in [0, 10, 100, 500, 1000, 1500, 2000]:

# 		plt.plot(parameters_tensor["log_Xi"][iteration,community,:], color = "blue", alpha = min(1, 10/(1.5 + 20 - iteration/100)))
	
# 	plt.plot(parameters["log_Xi"][community,:], color = "red")
# 	plt.show()

# 	for iteration in [0, 10, 100, 200, 300, 400, 500, 600, 800, 1000]:

# 		plt.plot(parameters_tensor["log_Xi"][iteration,1,:], color = "blue", alpha = min(1, 5/(1.5 + 10 - iteration/100)))
	
# 	plt.plot(parameters["log_Xi"][1,:], color = "red")
# 	plt.show()


# 	print("END")

	
# if __name__ == "__main__":
# 	import time

# 	from model_overdispersed import *
# 	from scipy.spatial.distance import pdist, squareform

# 	import matplotlib.pyplot as plt

# 	N_pop = 1000

# 	centroids  = np.load("Data/SpatialInference/Input/centroids.npy")
# 	city_index = np.load("Data/SpatialInference/Input/city_index.npy")

# 	index             = tf.convert_to_tensor(np.load("Data/SpatialInference/Input/reshuffle.npy")[:N_pop], dtype = tf.int32)
# 	location_sample   = tf.convert_to_tensor(np.load("Data/SpatialInference/Input/locations.npy"), dtype = tf.float32)
# 	covariates_sample = tf.convert_to_tensor(np.load("Data/SpatialInference/Input/covariates.npy"), dtype = tf.float32)

# 	covariates  = tf.gather(covariates_sample, index, axis = 0)
# 	locations  = tf.gather(location_sample, index, axis = 0)
# 	communities = tf.gather(city_index, index, axis = 0)

# 	mean_distances = np.zeros(np.unique(communities).shape[0])
# 	for i in range(np.unique(communities).shape[0]):
		
# 		geo_loc = tf.gather( locations, np.where(communities[:,0] == np.unique(communities)[i])[0], axis = 0).numpy()
# 		distances = pdist(geo_loc)
# 		mean_distances[i] = distances.mean()

# 	mean_distances = tf.expand_dims(tf.convert_to_tensor(mean_distances, dtype = tf.float32), axis = 1)

# 	city_covariates = tf.concat((centroids, mean_distances), axis = -1)

# 	_, populations = np.unique(communities, return_counts=True)
	
# 	observed_index = tf.concat((tf.ones(500), tf.zeros(500)), axis = 0)

# 	parameters = {"prior_infection":tf.convert_to_tensor([1-0.5, 0.5, 0.0], dtype = tf.float32),
# 		"log_beta":tf.math.log(
# 			tf.convert_to_tensor([3.0], dtype = tf.float32)),
# 		"b_S":tf.convert_to_tensor([+0.5], dtype = tf.float32),
# 		"b_I":tf.convert_to_tensor([+1.0], dtype = tf.float32),
# 		"log_gamma":tf.math.log(
# 			tf.convert_to_tensor([0.05], dtype = tf.float32)),
# 		"b_R":tf.convert_to_tensor([-0.1], dtype = tf.float32),
# 		"log_phi":tf.math.log(
# 			tf.convert_to_tensor([1.5], dtype = tf.float32)),
# 		"log_epsilon":tf.math.log(
# 			tf.convert_to_tensor([0.0001], dtype = tf.float32)),
# 		"logit_prob_testing":logit(
# 			tf.convert_to_tensor([0.1, 0.2, 0.5], dtype = tf.float32))}
	
# 	T = 100

# 	ovd_prior = {"log_Xi": tfp.distributions.Normal(loc = 0.0, scale = 0.5)}
	
# 	ovd_SIR = ovd_sbm_SIR(city_covariates, communities, covariates, observed_index, ovd_prior)

# 	for key in ovd_SIR.ovd_prior_shape.keys():
# 		parameters[key] = ovd_SIR.ovd_prior[key].sample((ovd_SIR.ovd_prior_shape[key], T))

# 	ovd_X_noisy, ovd_Y_noisy = ovd_simulator(ovd_SIR, parameters, T)
# 	plt.plot(tf.reduce_sum(ovd_X_noisy, axis = 2)[:,0,1], color = "purple")
# 	plt.show()

# 	plt.plot(tf.reduce_sum(ovd_Y_noisy, axis = 2)[:,0,2], color = "purple")
# 	plt.show()

# 	par_to_upd = parameters.copy()
# 	par_to_upd["log_Xi"] = ovd_SIR.ovd_prior["log_Xi"].sample((ovd_SIR.ovd_prior_shape["log_Xi"], T))

# 	learning_parameters = {"log_Xi":(ovd_SIR.ovd_prior_shape["log_Xi"], T)} 

# 	n_gradient_steps = 2000

# 	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# 	# ovd_CAL_prior(ovd_SIR, parameters)
# 	# N_pop*ovd_CAL_loss(ovd_SIR, parameters, ovd_Y_noisy[:,0,...])

# 	def blocking_function(ovd_ibm, log_likelihood_increment):

# 		return tf.einsum("...n,nc->...c", log_likelihood_increment, ovd_ibm.communities)
	
# 	def unblocking_function(ovd_ibm, indeces):

# 		return tf.einsum("...c,nc->...n", indeces, tf.cast(ovd_ibm.communities, tf.int32))

# 	parameters_SMC, log_likelihood = ovd_CALSMC(ovd_SIR, parameters, ovd_Y_noisy[:,0,...], blocking_function, unblocking_function, 1000)

# 	# loss_tensor, parameters_tensor = ovd_CAL_inference(ovd_SIR, par_to_upd, ovd_Y_noisy[:10,0,...], learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

# 	SMC_quantiles = np.quantile(parameters_SMC, (0.025, 0.975), axis = 1)

# 	community = 0
# 	plt.fill_between(np.linspace(1, T, T), SMC_quantiles[0,:,community], SMC_quantiles[1,:,community])
# 	plt.plot(parameters["log_Xi"][community,:], color = "red")
# 	plt.show()

# 	plt.plot(loss_tensor)
# 	plt.axhline(y = ovd_CAL_loss(ovd_SIR, parameters, ovd_Y_noisy[:,0,...]), color = "red")

# 	community = 0
# 	for iteration in [0, 10, 100, 500, 1000, 1500, 2000]:

# 		plt.plot(parameters_tensor["log_Xi"][iteration,community,:], color = "blue", alpha = min(1, 10/(1.5 + 20 - iteration/100)))
	
# 	plt.plot(parameters["log_Xi"][community,:], color = "red")
# 	plt.show()

# 	for iteration in [0, 10, 100, 200, 300, 400, 500, 600, 800, 1000]:

# 		plt.plot(parameters_tensor["log_Xi"][iteration,1,:], color = "blue", alpha = min(1, 5/(1.5 + 10 - iteration/100)))
	
# 	plt.plot(parameters["log_Xi"][1,:], color = "red")
# 	plt.show()


# 	print("END")
