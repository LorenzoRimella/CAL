import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

def bapf(ibm, parameters, y, P=128, seed_smc = None):
    
	T = tf.shape(y)[0]
	
	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

	pi_0 = ibm.pi_0(parameters)

	M = tf.shape(pi_0)[-1]
	X_p0 = tf.one_hot(tfp.distributions.Categorical(probs = pi_0).sample(P, seed = seed_smc_run[0]), M)

	log_likelihood = tf.zeros(1)

	def cond(input, t):

		return t<T

	def body(input, t):

		X_ptm1, log_likelihood = input

		seed_smc_pred, seed_smc_upda = tfp.random.split_seed( seed_smc_run[t], n = 2, salt = "seed_smc_body")

		k_x_tm1 = ibm.K_x(parameters, X_ptm1)
		g_t = ibm.G_t(parameters)

		p_X_tm1 = tf.einsum("PnK,PnKM->PnM", X_ptm1, k_x_tm1)
		p_Y_t   = tf.einsum("nMR,nR->nM", g_t, y[t,...])

		weights_np = tf.einsum("nM,PnM->Pn", p_Y_t, p_X_tm1)

		proposal_t = tf.einsum("PnM,Pn->PnM", tf.einsum("nM,PnM->PnM", p_Y_t, p_X_tm1), 1/weights_np)

		X_pt = tf.one_hot(tfp.distributions.Categorical(probs = proposal_t).sample(seed = seed_smc_pred), M)

		log_weights_pn = tf.math.log(tf.transpose(weights_np))
		shifted_weights_pn = tf.exp(log_weights_pn-tf.reduce_max(log_weights_pn, axis = 1, keepdims = True))
		norm_weights_pn = shifted_weights_pn/tf.reduce_sum(shifted_weights_pn, axis = 1, keepdims=True)

		# tf.reduce_sum(tf.math.log(tf.reduce_mean(weights_np, axis =0)))
		log_likelihood_increment = tf.reduce_sum(tf.math.log(tf.reduce_mean(shifted_weights_pn, axis =1)) + tf.reduce_max(log_weights_pn, axis = 1))

		indeces = tfp.distributions.Categorical(probs = norm_weights_pn).sample(P, seed = seed_smc_upda)
		res_X_pt = tf.transpose(tf.gather(tf.transpose(X_pt, [1, 0, 2 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2 ])

		return (res_X_pt, log_likelihood+log_likelihood_increment), t+1

	output = tf.while_loop(cond, body, loop_vars = ((X_p0, log_likelihood), 0))

	return output[0][1]

def batched_resampling(X_p, norm_weights, P, batch_size, seed_res):
    
    N = tf.shape(norm_weights)[0]
    M = tf.shape(X_p)[-1]

    batch_number = tf.cast(tf.math.floor(N/batch_size), tf.int32)

    seed_res_batched = tfp.random.split_seed( seed_res, n = batch_number, salt = "seed_smc_body_batched")

    def body(input, i):

        seed_res_batched_sub = tfp.random.split_seed( seed_res_batched[i], n = 4, salt = "sub_seed_smc_body_batched")
	
        sub_P = tf.cast(P/4, tf.int32)
        
        indeces_1 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[0])
        indeces_2 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[1])
        indeces_3 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[2])
        indeces_4 = tfp.distributions.Categorical(probs=norm_weights[(i*batch_size):((i*batch_size) + batch_size),:]).sample(sub_P, seed = seed_res_batched_sub[3])
        
        indeces = tf.concat((indeces_1, indeces_2, indeces_3, indeces_4), axis = 0)

        return tf.transpose(tf.gather(tf.transpose(X_p[:,(i*batch_size):((i*batch_size) + batch_size),:], [1, 0, 2]), tf.transpose(indeces), axis = 1, batch_dims = 1 ), [1, 0, 2])
    
    output = tf.scan(body, tf.range(0, batch_number, dtype=tf.int64), initializer = tf.zeros((P, batch_size, M)))
    output_transposed = tf.transpose(output, perm=[1, 0, 2, 3])

    output_reshaped = tf.reshape(output_transposed, [P, N, M])

    return output_reshaped

def batch_bapf(ibm, parameters, y, P = 1024, batch_size = 10, seed_smc = None):
    
	T = tf.shape(y)[0]

	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc_batched")

	pi_0 = ibm.pi_0(parameters)

	M = tf.shape(pi_0)[-1]
	X_p0 = tf.one_hot(tfp.distributions.Categorical(probs = pi_0).sample(P, seed = seed_smc_run[0]), M)

	log_likelihood = tf.zeros(1)

	def cond(input, t):

		return t<T

	def body(input, t):

		X_ptm1, log_likelihood = input

		seed_smc_pred, seed_smc_upda = tfp.random.split_seed( seed_smc_run[t], n = 2, salt = "seed_smc_body")

		k_x_tm1 = ibm.K_x(parameters, X_ptm1)
		g_t = ibm.G_t(parameters)

		p_X_tm1 = tf.einsum("PnK,PnKM->PnM", X_ptm1, k_x_tm1)
		p_Y_t   = tf.einsum("nMR,nR->nM", g_t, y[t,...])

		weights_np = tf.einsum("nM,PnM->Pn", p_Y_t, p_X_tm1)

		proposal_t = tf.einsum("PnM,Pn->PnM", tf.einsum("nM,PnM->PnM", p_Y_t, p_X_tm1), 1/weights_np)

		X_pt = tf.one_hot(tfp.distributions.Categorical(probs = proposal_t).sample(seed = seed_smc_pred), M)

		log_weights_pn = tf.math.log(tf.transpose(weights_np))
		shifted_weights_pn = tf.exp(log_weights_pn-tf.reduce_max(log_weights_pn, axis = 1, keepdims = True))
		norm_weights_pn = shifted_weights_pn/tf.reduce_sum(shifted_weights_pn, axis = 1, keepdims=True)

		# tf.reduce_sum(tf.math.log(tf.reduce_mean(weights_np, axis =0)))
		log_likelihood_increment = tf.reduce_sum(tf.math.log(tf.reduce_mean(shifted_weights_pn, axis =1)) + tf.reduce_max(log_weights_pn, axis = 1))

		res_X_pt = batched_resampling(X_pt, norm_weights_pn, P, batch_size, seed_smc_upda)

		return (res_X_pt, log_likelihood+log_likelihood_increment), t+1

	output = tf.while_loop(cond, body, loop_vars = ((X_p0, log_likelihood), 0))

	return output[0][1]


# if __name__ == "__main__":
# 	import numpy as np
# 	import tensorflow as tf
# 	import tensorflow_probability as tfp

# 	import os
# 	import sys
# 	sys.path.append('Scripts/')
# 	from model import *
# 	from CAL import *

# 	replicates = 100

# 	input_path  = "CAL/Data/old/Likelihood/Input/"
# 	output_path = "CAL/Data/old/Likelihood/SIS/"

# 	import time

# 	########################################
# 	# SIS
# 	M = 2
# 	N = 1000
# 	covmean = 0
# 	initial_infection_rate = 0.01
# 	T = 100

# 	covariates = tf.convert_to_tensor(np.load(input_path+"W_1000_numpy.npy"), dtype=tf.float32)
# 	y = tf.convert_to_tensor(np.einsum("ijk->kij", np.load(input_path+"Y_1000_numpy.npy")), dtype=tf.float32)
# 	Y_SIS = tf.concat((tf.expand_dims(tf.where(tf.reduce_sum(y[1:,...], axis = -1)==0, tf.ones(1, dtype = tf.float32), tf.zeros(1, dtype = tf.float32)), axis = -1), y[1:,...]), axis = -1)

# 	N = tf.shape(covariates)[0]

# 	parameters = {"beta_0": tf.convert_to_tensor([-np.log((1/initial_infection_rate)-1), +0], dtype = tf.float32),
# 		"beta_l": tf.convert_to_tensor([-1.0, +2.0], dtype = tf.float32),
# 		"beta_g": tf.convert_to_tensor([-1.0, -1.0], dtype = tf.float32),
# 		"iota": tf.math.log(tf.convert_to_tensor(0.001, dtype = tf.float32 ) ),
# 		"q":  tf.convert_to_tensor([0.6, 0.4], dtype = tf.float32)
# 		}


# 	SIS = simba_SIS(covariates)

# 	batch_bapf(SIS, parameters, Y_SIS)