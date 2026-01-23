import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

@tf.function(jit_compile=True)
def compute_proposal(ibm, parameters, X_ptm1, y_t):

	k_x_tm1 = ibm.K_x(parameters, X_ptm1)
	g_t = ibm.G_t(parameters)

	p_x_ntm1 = tf.einsum("PnK,PnKM->PnM", X_ptm1, k_x_tm1)
	p_y_nt   = tf.einsum("nMR,nR->nM", g_t, y_t)

	p_y_nt_given_x_ntm1 = tf.einsum("nM,PnM->Pn", p_y_nt, p_x_ntm1)

	proposal_nt = tf.einsum("PnM,Pn->PnM", tf.einsum("nM,PnM->PnM", p_y_nt, p_x_ntm1), 1/p_y_nt_given_x_ntm1)

	return proposal_nt, p_y_nt_given_x_ntm1

@tf.function(jit_compile=True)
def normalization_from_log(logw):

	norm_w = tf.math.exp(logw - tf.reduce_max(logw, axis = 0, keepdims=True))
	
	return norm_w/tf.reduce_sum(norm_w, axis = 0, keepdims = True)

@tf.function(jit_compile=True)
def compute_log_likelihood_increment(logw_pt):

	shifted_weights_p = tf.exp(logw_pt-tf.reduce_max(logw_pt, axis = 0, keepdims = True))

	return tf.math.log(tf.reduce_mean(shifted_weights_p, axis =0)) + tf.reduce_max(logw_pt, axis = 0)


def apf(ibm, parameters, y, P=256, seed_smc = None):
    
	T = tf.shape(y)[0]
	
	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

	pi_0 = ibm.pi_0(parameters)

	M = tf.shape(pi_0)[-1]
	X_p0 = tf.one_hot(tfp.distributions.Categorical(probs = pi_0).sample(P, seed = seed_smc_run[0]), M)
	logw_p0 = tf.zeros(P)

	log_likelihood = tf.zeros(1)

	def cond(input, t):

		return t<T

	def body(input, t):

		X_ptm1, logw_ptm1, log_likelihood = input

		seed_smc_pred, seed_smc_upda = tfp.random.split_seed( seed_smc_run[t], n = 2, salt = "seed_smc_body")

		# norm_w_ptm1 = tf.math.exp(logw_ptm1 - tf.reduce_max(logw_ptm1, axis = 0, keepdims=True))
		barw_ptm1 = normalization_from_log(logw_ptm1) #norm_w_ptm1/tf.reduce_sum(norm_w_ptm1, axis = 0, keepdims = True)

		proposal_nt, p_y_nt_given_x_ntm1 = compute_proposal(ibm, parameters, X_ptm1, y[t,...])

		logp_Y_t_given_X_tm1 = tf.math.reduce_sum(tf.math.log(p_y_nt_given_x_ntm1), axis = 1)
		# normp_Y_t_given_X_tm1 = tf.math.exp(logp_Y_t_given_X_tm1-tf.reduce_max(logp_Y_t_given_X_tm1, axis = 0))
		r_t = normalization_from_log(logp_Y_t_given_X_tm1) # normp_Y_t_given_X_tm1/tf.reduce_sum(normp_Y_t_given_X_tm1, axis = 0, keepdims = True)

		indeces = tfp.distributions.Categorical(probs = r_t).sample(P, seed = seed_smc_upda)
		res_proposal_nt = tf.gather(proposal_nt, indeces, axis = 0)
		res_barw_ptm1  = tf.gather(barw_ptm1, indeces, axis = 0)
		res_r_t        = tf.gather(r_t, indeces, axis = 0)
		res_logp_Y_t_given_X_tm1 = tf.gather(logp_Y_t_given_X_tm1, indeces, axis = 0)

		tildelogw_ptm1 = tf.math.log(res_barw_ptm1)-tf.math.log(res_r_t)

		X_pt = tf.one_hot(tfp.distributions.Categorical(probs = res_proposal_nt).sample(seed = seed_smc_pred), M)

		logw_pt = tildelogw_ptm1 + res_logp_Y_t_given_X_tm1

		# resp_x_ntm1 = tf.einsum("PnK,PnKM->PnM", tf.gather(X_ptm1, indeces, axis = 0), ibm.K_x(parameters, tf.gather(X_ptm1, indeces, axis = 0)))
		# logw_pt_1 = tildelogw_ptm1 + tf.reduce_sum(tf.math.log(tf.einsum("PnM,PnM->Pn", tf.einsum("nM,PnM->PnM", p_y_nt, resp_x_ntm1), X_pt)), axis =1) - tf.reduce_sum(tf.math.log(tf.einsum("PnM,PnM->Pn", res_proposal_nt, X_pt)), axis =1)
		# logw_pt_3 = tf.math.log(tf.reduce_mean(tf.math.exp(logp_Y_t_given_X_tm1-tf.reduce_max(logp_Y_t_given_X_tm1))))+tf.reduce_max(logp_Y_t_given_X_tm1)

		log_likelihood_increment = compute_log_likelihood_increment(logw_pt)

		return (X_pt, logw_pt, log_likelihood+log_likelihood_increment), t+1

	output = tf.while_loop(cond, body, loop_vars = ((X_p0, logw_p0, log_likelihood), 0))

	return output[0][2]

# if __name__ == "__main__":
# 	import numpy as np
# 	import tensorflow as tf
# 	import tensorflow_probability as tfp

# 	import os
# 	import sys
# 	sys.path.append('CAL/Scripts/')
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

# 	tf.random.set_seed((123))
# 	np.random.seed((123))

# 	SIS = simba_SIS(covariates)

# 	out1 = apf(SIS, parameters, Y_SIS)
# 	out2 = deg_apf(SIS, parameters, Y_SIS)

# 	print("ciao")