import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from smc import compute_proposal, normalization_from_log, compute_log_likelihood_increment

def bapf(ibm, parameters, y, P=128, seed_smc = None):
    
	T = tf.shape(y)[0]
	
	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

	pi_0 = ibm.pi_0(parameters)

	M = tf.shape(pi_0)[-1]
	X_p0 = tf.one_hot(tfp.distributions.Categorical(probs = pi_0).sample(P, seed = seed_smc_run[0]), M)
	logw_p0 = tf.zeros((P, ibm.N))

	log_likelihood = tf.zeros(1)

	def cond(input, t):

		return t<T

	def body(input, t):

		X_ptm1, logw_ptm1, log_likelihood = input

		seed_smc_pred, seed_smc_upda = tfp.random.split_seed( seed_smc_run[t], n = 2, salt = "seed_smc_body")

		barw_ptm1 = normalization_from_log(logw_ptm1)

		proposal_nt, p_y_nt_given_x_ntm1 = compute_proposal(ibm, parameters, X_ptm1, y[t,...])

		r_nt = normalization_from_log(tf.math.log(p_y_nt_given_x_ntm1)) 

		indeces = tfp.distributions.Categorical(probs = tf.transpose(r_nt)).sample(P, seed = seed_smc_upda)
		res_proposal_nt = tf.transpose(tf.gather(tf.transpose(proposal_nt, [1, 0, 2 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2 ])
		res_barw_ptm1   = tf.transpose(tf.gather(tf.transpose(barw_ptm1, [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
		res_r_nt        = tf.transpose(tf.gather(tf.transpose(r_nt, [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
		res_p_y_nt_given_x_ntm1 = tf.transpose(tf.gather(tf.transpose(p_y_nt_given_x_ntm1, [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])

		tildelogw_ptm1 = tf.math.log(res_barw_ptm1)-tf.math.log(res_r_nt)

		X_pt = tf.one_hot(tfp.distributions.Categorical(probs = res_proposal_nt).sample(seed = seed_smc_pred), M)

		logw_pt = tildelogw_ptm1 + tf.math.log(res_p_y_nt_given_x_ntm1)

		log_likelihood_increment = compute_log_likelihood_increment(tf.reduce_sum(logw_pt, axis = 1))

		return (X_pt, logw_pt, log_likelihood+log_likelihood_increment), t+1

	output = tf.while_loop(cond, body, loop_vars = ((X_p0, logw_p0, log_likelihood), 0))

	return output[0][2]

def batched_resampling(proposal_nt, barw_ptm1, r_nt, p_y_nt_given_x_ntm1, norm_weights, P, batch_size, seed_res):
    
	N = tf.shape(norm_weights)[0]
	M = tf.shape(proposal_nt)[-1]

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
		res_proposal_nt = tf.transpose(tf.gather(tf.transpose(proposal_nt[:,(i*batch_size):((i*batch_size) + batch_size),:], [1, 0, 2 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0, 2 ])
		res_barw_ptm1   = tf.transpose(tf.gather(tf.transpose(barw_ptm1[:,(i*batch_size):((i*batch_size) + batch_size)], [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
		res_r_nt        = tf.transpose(tf.gather(tf.transpose(r_nt[:,(i*batch_size):((i*batch_size) + batch_size)], [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])
		res_p_y_nt_given_x_ntm1 = tf.transpose(tf.gather(tf.transpose(p_y_nt_given_x_ntm1[:,(i*batch_size):((i*batch_size) + batch_size)], [1, 0 ]), tf.transpose(indeces), axis = 1, batch_dims=1 ), [1, 0 ])

		return res_proposal_nt, res_barw_ptm1, res_r_nt, res_p_y_nt_given_x_ntm1
    
	tuple_output = tf.scan(body, tf.range(0, batch_number, dtype=tf.int64), initializer = (tf.zeros((P, batch_size, M)), tf.zeros((P, batch_size)), tf.zeros((P, batch_size)), tf.zeros((P, batch_size))))
	out_transpose_res_proposal_nt         = tf.transpose(tuple_output[0], perm=[1, 0, 2, 3])
	out_transpose_res_barw_ptm1           = tf.transpose(tuple_output[1], perm=[1, 0, 2])
	out_transpose_res_r_nt                = tf.transpose(tuple_output[2], perm=[1, 0, 2])
	out_transpose_res_p_y_nt_given_x_ntm1 = tf.transpose(tuple_output[3], perm=[1, 0, 2])
	
	out_reshape_res_proposal_nt         = tf.reshape(out_transpose_res_proposal_nt, [P, N, M])
	out_reshape_res_barw_ptm1           = tf.reshape(out_transpose_res_barw_ptm1, [P, N])
	out_reshape_res_r_nt                = tf.reshape(out_transpose_res_r_nt, [P, N])
	out_reshape_res_p_y_nt_given_x_ntm1 = tf.reshape(out_transpose_res_p_y_nt_given_x_ntm1, [P, N])

	return out_reshape_res_proposal_nt, out_reshape_res_barw_ptm1, out_reshape_res_r_nt, out_reshape_res_p_y_nt_given_x_ntm1

def batch_bapf(ibm, parameters, y, P=128, batch_size = 10, seed_smc = None):
    
	T = tf.shape(y)[0]
	
	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

	pi_0 = ibm.pi_0(parameters)

	M = tf.shape(pi_0)[-1]
	X_p0 = tf.one_hot(tfp.distributions.Categorical(probs = pi_0).sample(P, seed = seed_smc_run[0]), M)
	logw_p0 = tf.zeros((P, ibm.N))

	log_likelihood = tf.zeros(1)

	def cond(input, t):

		return t<T

	def body(input, t):

		X_ptm1, logw_ptm1, log_likelihood = input

		seed_smc_pred, seed_smc_upda = tfp.random.split_seed( seed_smc_run[t], n = 2, salt = "seed_smc_body")

		barw_ptm1 = normalization_from_log(logw_ptm1)

		proposal_nt, p_y_nt_given_x_ntm1 = compute_proposal(ibm, parameters, X_ptm1, y[t,...])

		r_nt = normalization_from_log(tf.math.log(p_y_nt_given_x_ntm1)) 

		res_proposal_nt, res_barw_ptm1, res_r_nt, res_p_y_nt_given_x_ntm1 = batched_resampling(proposal_nt, barw_ptm1, r_nt, p_y_nt_given_x_ntm1, tf.transpose(r_nt), P, batch_size, seed_smc_upda)

		tildelogw_ptm1 = tf.math.log(res_barw_ptm1)-tf.math.log(res_r_nt)

		X_pt = tf.one_hot(tfp.distributions.Categorical(probs = res_proposal_nt).sample(seed = seed_smc_pred), M)

		logw_pt = tildelogw_ptm1 + tf.math.log(res_p_y_nt_given_x_ntm1)

		log_likelihood_increment = compute_log_likelihood_increment(tf.reduce_sum(logw_pt, axis = 1))

		return (X_pt, logw_pt, log_likelihood+log_likelihood_increment), t+1

	output = tf.while_loop(cond, body, loop_vars = ((X_p0, logw_p0, log_likelihood), 0))

	return output[0][2]

# if __name__ == "__main__":
# 	import numpy as np
# 	import tensorflow as tf
# 	import tensorflow_probability as tfp

# 	import os
# 	import sys
# 	sys.path.append('Scripts/')
# 	from model import *
# 	from CAL import *

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


# 	tf.random.set_seed((123))
# 	np.random.seed((123))

# 	out1 = deg_bapf(SIS, parameters, Y_SIS)
# 	# bout1 = deg_batch_bapf(SIS, parameters, Y_SIS)
	
# 	out2 = bapf(SIS, parameters, Y_SIS)
# 	bout2 = batch_bapf(SIS, parameters, Y_SIS)

# 	print("ciao")