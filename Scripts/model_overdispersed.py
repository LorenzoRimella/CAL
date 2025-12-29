import numpy as np

from model import *

import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd

# @tf.function(jit_compile=True)
def ovd_simulator(ibm, parameters, T,  sample_size = 1, seed_sim = None):
	"""An individual based model simulator without extra state

	It produces a simulation form the individual based model given in input.

	Args:
	ibm: an individual based model object, meaning that it require the 
	following methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	parameters: a dictionary with the parameters as keys. This has
	to be build to be compatible with ibm
	seed_sim: Floating point tensor with shape (2) representing
	the given seed
	"""

	seed_0, seed_carry = tfp.random.split_seed( seed_sim, n = 2, salt='simulator')

	pi_0 = ibm.pi_0(parameters)
	x_0  = simulation_step_0(pi_0, sample_size, seed_sim = seed_0)
	y_0 = tf.zeros(tf.shape(x_0), dtype = tf.float32)
	y_0 = tf.concat((y_0, y_0[...,-1:]), axis = -1)

	def body(input, t):

		x_tm1, _, seed_carry = input

		seed_dynamic, seed_emission, seed_carry = tfp.random.split_seed( seed_carry, n=3, salt='simulator_step_t')

		k_x_tm1 = ibm.K_x(parameters, x_tm1, t)
		x_t = dynamic_t(x_tm1, k_x_tm1, seed_dynamic)

		g_t = ibm.G_t(parameters, t)
		y_t = emission_t(x_t, g_t, seed_emission)

		return x_t, y_t, seed_carry

	X, Y, _ = tf.scan(body, tf.range(1, T+1), initializer = (x_0, y_0, seed_carry))

	return tf.concat((tf.expand_dims(x_0, axis = 0), X), axis = 0), Y

# OVERDISPERSED MODELS

class ovd_logistic_SIS():
	"""A logistic SIS 

	It has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, covariates, ovd_prior):
		super().__init__()
		"""Construct the basic SIS

		This creates some quantities that do not change

		Args:
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.covariates = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 2

		self.ovd_prior = ovd_prior
		self.ovd_prior_shape = {"log_Xi":1} #np.unique(communities).shape[0]}

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x, t):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""

		N_float     = tf.cast(self.N, dtype = tf.float32)
		infectivity = tf.math.exp(tf.einsum("nm,m->n", self.covariates, parameters["b_I"]))

		# xi_t = tf.math.exp(parameters["log_Xi"][...,0,t-1])
		xi_t = tf.einsum("...bt,t->...", tf.math.exp(parameters["log_Xi"]), tf.one_hot(t-1, tf.shape(parameters["log_Xi"])[-1]))

		infectious_pressure = tf.einsum("...n,n->...", x[...,1], infectivity)/N_float
		susceptibility      = tf.math.exp(parameters["log_beta"]  + tf.einsum("nm,m->n", self.covariates, parameters["b_S"]))
		susceptibility      = tf.einsum("...,...n->...n", xi_t, susceptibility)
		recoverability      = tf.math.exp(parameters["log_gamma"] + tf.einsum("nm,m->n", self.covariates, parameters["b_R"]))

		K_t_n_SI = 1 - tf.math.exp(-tf.einsum("...,...n->...n", infectious_pressure, susceptibility))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = 1 - tf.math.exp(-tf.einsum("...,n->...n", tf.ones(tf.shape(infectious_pressure)), recoverability))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters, t):
		"""
		The stochastic matrix computing the reporting probabilities.
		Each row refer to a different state, while the columns refer
		to the reporting states, with the first one being unreported

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		prob_testing = tf.expand_dims(tf.math.sigmoid(parameters["logit_prob_testing"]), axis =0)*tf.ones((self.N, self.M))
		prob_nontesting =  1- prob_testing

		prob_SS_IS = tf.stack((tf.math.sigmoid(parameters["logit_specificity"]), 
					1 - tf.math.sigmoid(parameters["logit_sensitivity"])), axis = 1)*tf.ones((self.N, self.M))

		return tf.stack((prob_nontesting, 
					prob_SS_IS*prob_testing, 
					(1 - prob_SS_IS)*prob_testing), axis = -1)


class ovd_sbm_SIR():
	"""A stochastic block model SIS overdispersed 

	We implement a stochastic block model where each individual has a
	community and some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, city_covariates, communities, covariates, observed_index, ovd_prior):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.communities     = tf.one_hot(indices = tf.cast(communities[:,0], tf.int32), depth = tf.cast(tf.shape(city_covariates)[0], tf.int32), dtype = tf.float32)
		self.city_covariates = city_covariates
		self.covariates      = covariates
		self.observed_index = observed_index

		self.N = tf.shape(covariates)[0]
		self.M = 3

		self.ovd_prior = ovd_prior
		self.ovd_prior_shape = {"log_Xi":1} #np.unique(communities).shape[0]}

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		initial_infected_centroid_index = tf.reduce_all(tf.stack((self.city_covariates[:,0]<tf.reduce_max(self.city_covariates[:,0])/2, 
						   		 self.city_covariates[:,1]>tf.reduce_max(self.city_covariates[:,1])*0.8), axis = -1), axis = -1)
		
		initial_infected_centroid_index = tf.cast(initial_infected_centroid_index, dtype = tf.float32)

		initial_infected_index = tf.einsum("...c,nc->...n", initial_infected_centroid_index, self.communities)
		
		mask = tf.expand_dims(initial_infected_index, axis = -1)

		prior_infection_1 = tf.stack((tf.ones(self.N), tf.zeros(self.N), tf.zeros(self.N)), axis = -1)
		prior_infection_2 = parameters["prior_infection"]*tf.ones((self.N, self.M))

		return mask*prior_infection_2 + (1-mask)*prior_infection_1
	
	def B_matrix(self, parameters):

		float_centroids = tf.cast(self.city_covariates, dtype = tf.float32)
		
		d_1 = tf.math.pow(float_centroids[:,0:1] - tf.transpose(float_centroids[:,0:1]), 2)
		d_2 = tf.math.pow(float_centroids[:,1:2] - tf.transpose(float_centroids[:,1:2]), 2)
		d = tf.math.sqrt(d_1 + d_2)

		mask = tf.linalg.diag(tf.ones(tf.shape(self.city_covariates[:,2])))
		d    = (1-mask)*d + tf.linalg.diag(self.city_covariates[:,2]) 

		B = tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_phi"])).prob(d)

		return B
    
	def K_x(self, parameters, x, t):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
		
		N_float = tf.cast(self.N, dtype = tf.float32)

		xi_t = tf.math.exp(parameters["log_Xi"][...,t-1])
		# community_xi_t = tf.einsum("...c,...nc->...n", xi_t, self.communities)
		community_xi_t = xi_t #tf.expand_dims(xi_t, axis = 1)

		infectivity = tf.math.exp(tf.einsum("nm,m->n", self.covariates, parameters["b_I"]))
		I_c         = tf.einsum("...n,nc->...c", infectivity*x[...,1], self.communities)
		infectious_pressure = tf.einsum("ij,...j->...i", self.B_matrix(parameters), I_c)/N_float
		infectious_pressure = tf.einsum("...c,nc->...n", infectious_pressure, self.communities)

		susceptibility      = tf.math.exp( parameters["log_beta"]  + tf.einsum("nm,m->n", self.covariates, parameters["b_S"]))
		recoverability      = tf.math.exp( parameters["log_gamma"] + tf.einsum("nm,m->n", self.covariates, parameters["b_R"]))

		K_t_n_SI =  1 - tf.math.exp(-community_xi_t*tf.einsum("...n,n->...n", infectious_pressure, susceptibility) - tf.math.exp(parameters["log_epsilon"]))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI, tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IR = 1 - tf.math.exp(-tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), recoverability))
		K_t_n_IR = tf.expand_dims(K_t_n_IR, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), 1 - K_t_n_IR, K_t_n_IR), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), tf.zeros(tf.shape(K_t_n_IR)), tf.ones(tf.shape(K_t_n_IR))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_R), axis  = -2)
	
	def G_t(self, parameters, t):
		"""
		The stochastic matrix computing the reporting probabilities.
		Each row refer to a different state, while the columns refer
		to the reporting states, with the first one being unreported

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		prob_testing = tf.linalg.diag(tf.math.sigmoid(parameters["logit_prob_testing"]))

		G_common     = tf.concat((tf.expand_dims(1 - tf.math.sigmoid(parameters["logit_prob_testing"]), axis = -1), prob_testing), axis = -1)
		G_unobserved = tf.concat((tf.expand_dims(tf.ones(tf.shape(parameters["logit_prob_testing"])), axis = -1), tf.zeros(tf.shape(prob_testing))), axis = -1)

		mask = tf.expand_dims(tf.expand_dims(self.observed_index, axis = -1), axis = -1)*tf.ones((self.N, self.M, self.M+1))

		return tf.expand_dims(G_common, axis = 0)*mask + tf.expand_dims(G_unobserved, axis = 0)*(1 - mask)

class ovd_FM_SIR():
	"""
	"""

	def __init__(self, local_autorities_covariates, communities, farms_covariates, ovd_prior):
		super().__init__()
		"""
		"""

		self.communities = tf.one_hot(indices = tf.cast(communities[:,0], tf.int32), depth = tf.cast(tf.shape(local_autorities_covariates)[0], tf.int32), dtype = tf.float32)
		self.local_autorities_covariates   = local_autorities_covariates
		self.farms_covariates  = farms_covariates

		self.N = tf.shape(farms_covariates)[0]
		self.M = 3

		self.ovd_prior = ovd_prior
		self.ovd_prior_shape = {"log_Xi": 1} #np.unique(communities).shape[0]}

	def pi_0(self, parameters):
		"""
		"""

		N_float = tf.cast(self.N, dtype = tf.float32)

		infectivity = tf.math.exp(tf.einsum("nm,m->n", self.farms_covariates, parameters["b_I"]))
		I_c         = tf.einsum("...n,nc->...c", infectivity*tf.ones(self.N, dtype = tf.float32)/N_float, self.communities)
		infectious_pressure = tf.einsum("ij,...j->...i", self.I_matrix(parameters), I_c)/N_float
		infectious_pressure = tf.einsum("...c,nc->...n", infectious_pressure, self.communities)

		susceptibility      = tf.math.exp( parameters["log_beta"]  + tf.einsum("nm,m->n", self.farms_covariates, parameters["b_S"]))

		pi_0_I =  1 - tf.math.exp(-tf.exp(parameters["log_tau"])*tf.einsum("...n,n->...n", infectious_pressure, susceptibility) - tf.math.exp(parameters["log_epsilon"]))

		return tf.stack(( 1- pi_0_I, pi_0_I, tf.zeros(tf.shape(pi_0_I)) ), axis = -1)
		
		# return parameters["prior_infection"]*tf.ones((self.N, self.M))

	def I_matrix(self, parameters):

		loc_aut_d_1 = tf.math.pow(self.local_autorities_covariates[:,0:1] - tf.transpose(self.local_autorities_covariates[:,0:1]), 2)
		loc_aut_d_2 = tf.math.pow(self.local_autorities_covariates[:,1:2] - tf.transpose(self.local_autorities_covariates[:,1:2]), 2)
		loc_aut_d = tf.math.sqrt(loc_aut_d_1 + loc_aut_d_2)

		mask = tf.linalg.diag(tf.ones(tf.shape(self.local_autorities_covariates[:,2])))
		loc_aut_d = loc_aut_d*(1 - mask) + tf.linalg.diag(self.local_autorities_covariates[:,2])

		return tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_phi"])).prob(loc_aut_d)
	
	def C_matrix(self, parameters):

		loc_aut_d_1 = tf.math.pow(self.local_autorities_covariates[:,0:1] - tf.transpose(self.local_autorities_covariates[:,0:1]), 2)
		loc_aut_d_2 = tf.math.pow(self.local_autorities_covariates[:,1:2] - tf.transpose(self.local_autorities_covariates[:,1:2]), 2)
		loc_aut_d = tf.math.sqrt(loc_aut_d_1 + loc_aut_d_2)

		mask = tf.linalg.diag(tf.ones(tf.shape(self.local_autorities_covariates[:,2])))
		loc_aut_d = loc_aut_d*(1 - mask) + tf.linalg.diag(self.local_autorities_covariates[:,2])

		return tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_psi"])).prob(loc_aut_d)
    
	def K_x(self, parameters, x, t):
		"""
		"""
		
		N_float = tf.cast(self.N, dtype = tf.float32)

		xi_t = tf.einsum("...t,t->...", tf.math.exp(parameters["log_Xi"]), tf.one_hot(t-1, tf.shape(parameters["log_Xi"])[-1]))

		infectivity = tf.math.exp(tf.einsum("nm,m->n", self.farms_covariates, parameters["b_I"]))
		I_c         = tf.einsum("...n,nc->...c", infectivity*x[...,1], self.communities)
		infectious_pressure = tf.einsum("ij,...j->...i", self.I_matrix(parameters), I_c)/N_float
		infectious_pressure = tf.einsum("...c,nc->...n", infectious_pressure, self.communities)

		susceptibility      = tf.math.exp( parameters["log_beta"] + tf.math.log(xi_t)  + tf.einsum("nm,m->n", self.farms_covariates, parameters["b_S"]))

		culling_pressure = tf.einsum("ij,...j->...i", self.C_matrix(parameters), I_c)/N_float
		culling_pressure = tf.exp(parameters["log_rho"])*tf.einsum("...c,nc->...n", culling_pressure, self.communities)

		prob_culling = (1 - tf.math.exp(-culling_pressure))

		K_t_n_SR = prob_culling
		K_t_n_SI = (1 - tf.math.exp(-tf.einsum("...n,...n->...n", infectious_pressure, susceptibility) - tf.math.exp(parameters["log_epsilon"])))*(1 - prob_culling)

		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_SR = tf.expand_dims(K_t_n_SR, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI-K_t_n_SR, K_t_n_SI, K_t_n_SR), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IR = prob_culling + (1 - prob_culling)*(1 - tf.math.exp(-tf.math.exp(parameters["log_gamma"])))
		K_t_n_IR = tf.expand_dims(K_t_n_IR, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), 1 - K_t_n_IR, K_t_n_IR), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), tf.zeros(tf.shape(K_t_n_IR)), tf.ones(tf.shape(K_t_n_IR))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_R), axis  = -2)
	
	def G_t(self, parameters, t):
		"""
		"""

		prob_I_test = tf.math.sigmoid(parameters["logit_prob_I_testing"])

		prob_testing = tf.concat((tf.zeros(tf.shape(prob_I_test)), prob_I_test, tf.zeros(tf.shape(prob_I_test))), axis = 0)

		G_common = tf.concat((tf.expand_dims(1 - prob_testing, axis = -1), tf.linalg.diag(prob_testing)), axis = -1)

		return tf.expand_dims(G_common, axis = 0)*tf.ones((self.N, self.M, self.M+1))	
	

class ovd_FM_SIR_communities():
	"""
	"""

	def __init__(self, local_autorities_covariates, communities, farms_covariates, ovd_prior):
		super().__init__()
		"""
		"""

		self.communities = tf.one_hot(indices = tf.cast(communities[:,0], tf.int32), depth = tf.cast(tf.shape(local_autorities_covariates)[0], tf.int32), dtype = tf.float32)
		self.local_autorities_covariates   = local_autorities_covariates
		self.farms_covariates  = farms_covariates

		self.N = tf.shape(farms_covariates)[0]
		self.M = 3

		self.ovd_prior = ovd_prior
		self.ovd_prior_shape = {"log_Xi": tf.shape(self.communities)[1]} #np.unique(communities).shape[0]}

	def pi_0(self, parameters):
		"""
		"""

		N_float = tf.cast(self.N, dtype = tf.float32)

		infectivity = tf.math.exp(tf.einsum("nm,m->n", self.farms_covariates, parameters["b_I"]))
		I_c         = tf.einsum("...n,nc->...c", infectivity*tf.ones(self.N, dtype = tf.float32)/N_float, self.communities)
		infectious_pressure = tf.einsum("ij,...j->...i", self.I_matrix(parameters), I_c)/N_float
		infectious_pressure = tf.einsum("...c,nc->...n", infectious_pressure, self.communities)

		susceptibility      = tf.math.exp( parameters["log_beta"]  + tf.einsum("nm,m->n", self.farms_covariates, parameters["b_S"]))

		pi_0_I =  1 - tf.math.exp(-tf.exp(parameters["log_tau"])*tf.einsum("...n,n->...n", infectious_pressure, susceptibility) - tf.math.exp(parameters["log_epsilon"]))

		return tf.stack(( 1- pi_0_I, pi_0_I, tf.zeros(tf.shape(pi_0_I)) ), axis = -1)
		
		# return parameters["prior_infection"]*tf.ones((self.N, self.M))

	def I_matrix(self, parameters):

		loc_aut_d_1 = tf.math.pow(self.local_autorities_covariates[:,0:1] - tf.transpose(self.local_autorities_covariates[:,0:1]), 2)
		loc_aut_d_2 = tf.math.pow(self.local_autorities_covariates[:,1:2] - tf.transpose(self.local_autorities_covariates[:,1:2]), 2)
		loc_aut_d = tf.math.sqrt(loc_aut_d_1 + loc_aut_d_2)

		mask = tf.linalg.diag(tf.ones(tf.shape(self.local_autorities_covariates[:,2])))
		loc_aut_d = loc_aut_d*(1 - mask) + tf.linalg.diag(self.local_autorities_covariates[:,2])

		return tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_phi"])).prob(loc_aut_d)
	
	def C_matrix(self, parameters):

		loc_aut_d_1 = tf.math.pow(self.local_autorities_covariates[:,0:1] - tf.transpose(self.local_autorities_covariates[:,0:1]), 2)
		loc_aut_d_2 = tf.math.pow(self.local_autorities_covariates[:,1:2] - tf.transpose(self.local_autorities_covariates[:,1:2]), 2)
		loc_aut_d = tf.math.sqrt(loc_aut_d_1 + loc_aut_d_2)

		mask = tf.linalg.diag(tf.ones(tf.shape(self.local_autorities_covariates[:,2])))
		loc_aut_d = loc_aut_d*(1 - mask) + tf.linalg.diag(self.local_autorities_covariates[:,2])

		return tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_psi"])).prob(loc_aut_d)
    
	def K_x(self, parameters, x, t):
		"""
		"""
		
		N_float = tf.cast(self.N, dtype = tf.float32)

		# xi_t = tf.math.exp(parameters["log_Xi"][...,t-1])
		# xi_t = tf.math.exp(parameters["log_Xi"][...,0,t-1])
		xi_t = tf.einsum("...t,t->...", tf.math.exp(parameters["log_Xi"]), tf.one_hot(t-1, tf.shape(parameters["log_Xi"])[-1]))

		infectivity = tf.math.exp(tf.einsum("nm,m->n", self.farms_covariates, parameters["b_I"]))
		I_c         = tf.einsum("...n,nc->...c", infectivity*x[...,1], self.communities)
		infectious_pressure = tf.einsum("ij,...j->...i", self.I_matrix(parameters), I_c)/N_float
		infectious_pressure = tf.einsum("...c,nc->...n", infectious_pressure, self.communities)

		xi_t_c = tf.einsum("...c,nc->...n", xi_t, self.communities)

		susceptibility      = tf.math.exp( parameters["log_beta"] + tf.math.log(xi_t_c)  + tf.einsum("nm,m->n", self.farms_covariates, parameters["b_S"]))

		culling_pressure = tf.einsum("ij,...j->...i", self.C_matrix(parameters), I_c)/N_float
		culling_pressure = tf.exp(parameters["log_rho"])*tf.einsum("...c,nc->...n", culling_pressure, self.communities)

		prob_culling = (1 - tf.math.exp(-culling_pressure))

		K_t_n_SR = prob_culling
		K_t_n_SI = (1 - tf.math.exp(-tf.einsum("...n,...n->...n", infectious_pressure, susceptibility) - tf.math.exp(parameters["log_epsilon"])))*(1 - prob_culling)

		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_SR = tf.expand_dims(K_t_n_SR, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI-K_t_n_SR, K_t_n_SI, K_t_n_SR), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IR = prob_culling + (1 - prob_culling)*(1 - tf.math.exp(-tf.math.exp(parameters["log_gamma"])))
		K_t_n_IR = tf.expand_dims(K_t_n_IR, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), 1 - K_t_n_IR, K_t_n_IR), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), tf.zeros(tf.shape(K_t_n_IR)), tf.ones(tf.shape(K_t_n_IR))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_R), axis  = -2)
	
	def G_t(self, parameters, t):
		"""
		"""

		prob_I_test = tf.math.sigmoid(parameters["logit_prob_I_testing"])

		prob_testing = tf.concat((tf.zeros(tf.shape(prob_I_test)), prob_I_test, tf.zeros(tf.shape(prob_I_test))), axis = 0)

		G_common = tf.concat((tf.expand_dims(1 - prob_testing, axis = -1), tf.linalg.diag(prob_testing)), axis = -1)

		return tf.expand_dims(G_common, axis = 0)*tf.ones((self.N, self.M, self.M+1))	


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
	
	# ovd_prior = {}
# 	ovd_prior["log_Xi"] = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
# 	parameters["log_Xi"] = ovd_prior.sample((tf.shape(local_autorities_covariates)[0], 100))
# 	ovdSIR = ovd_FM_SIR_communities(local_autorities_covariates, communities, farms_covariates, ovd_prior)

# 	ovd_X, ovd_Y = ovd_simulator(ovdSIR, parameters, 100)

# if __name__ == "__main__":
# 	import time

# 	from model import *
# 	from scipy.spatial.distance import pdist, squareform

# 	import matplotlib.pyplot as plt


# 	N = 30000

# 	local_autorities_covariates  = tf.convert_to_tensor(np.load("CAL/Data/FM/local_autorities_covariates.npy"), dtype = tf.float32)[:N,:]
# 	farms_covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/farms_covariates.npy"), dtype = tf.float32)[:N,:]

# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/cut_Y_FM.npy")[:,:N,:], dtype = tf.float32)

# 	communities = farms_covariates[:,-1:]
# 	farms_covariates = farms_covariates[:,:2]

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
	
# 	ovd_prior = tfp.distributions.Normal(loc = 0.0, scale = 1)
	
# 	ovdSIR = ovd_FM_SIR(local_autorities_covariates, communities, farms_covariates, ovd_prior)
	
# 	T = 100

# 	for i in range(5):
# 		parameters["log_Xi"] = ovd_prior.sample((1, T))
# 		ovd_X, ovd_Y = ovd_simulator(ovdSIR, parameters, T)
# 		plt.plot(tf.reduce_sum(ovd_X, axis = 2)[:,0,1], color = "orange")
	
# 	for i in range(5):
# 		parameters["log_Xi"] = tf.zeros((1, T))
# 		ovd_X, ovd_Y = ovd_simulator(ovdSIR, parameters, T)
# 		plt.plot(tf.reduce_sum(ovd_X, axis = 2)[:,0,1], color = "red")

# 	print("END")

# if __name__ == "__main__":
# 	import time

# 	from model import *
# 	from scipy.spatial.distance import pdist, squareform

# 	import matplotlib.pyplot as plt

# 	N_pop = 1000

# 	covariates_sample = tf.convert_to_tensor(np.load("Data/SpatialInference/Input/covariates.npy"), dtype = tf.float32)

# 	parameters = {"prior_infection":tf.convert_to_tensor([1-0.05, 0.05], dtype = tf.float32),
# 		"log_beta":tf.math.log(
# 			tf.convert_to_tensor([0.5], dtype = tf.float32)),
# 		"b_S":tf.convert_to_tensor([+0.5], dtype = tf.float32),
# 		"b_I":tf.convert_to_tensor([-0.5], dtype = tf.float32),
# 		"b_R":tf.convert_to_tensor([+1.0], dtype = tf.float32),
# 		"log_gamma":tf.math.log(
# 			tf.convert_to_tensor([0.5], dtype = tf.float32)),
# 		"logit_prob_testing":logit(
# 			tf.convert_to_tensor([0.1, 0.2], dtype = tf.float32)),
# 		"logit_specificity":logit( tf.convert_to_tensor([0.95], dtype = tf.float32)),
# 		"logit_sensitivity":logit( tf.convert_to_tensor([0.95], dtype = tf.float32)),
# 			}
# 	ovd_SIS = ovd_logistic_SIS(covariates_sample)
	
# 	T = 100

# 	prior = tfp.distributions.Normal(loc = 0.0, scale = 0.5)

# 	for i in range(5):
# 		parameters["log_Xi"] = prior.sample((T))
# 		ovd_X, ovd_Y = ovd_simulator(ovd_SIS, parameters, T)
# 		plt.plot(tf.reduce_sum(ovd_X, axis = 2)[:,0,1], color = "orange")
	
# 	for i in range(5):
# 		parameters["log_Xi"] = tf.zeros((T))
# 		ovd_X, ovd_Y = ovd_simulator(ovd_SIS, parameters, T)
# 		plt.plot(tf.reduce_sum(ovd_X, axis = 2)[:,0,1], color = "red")

# 	print("END")

	
# if __name__ == "__main__":
# 	import time

# 	from model import *
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
# 	SIR = sbm_SIR(city_covariates, communities, covariates, observed_index)
	
# 	T = 100

# 	for i in range(5):
# 		X, Y = simulator(SIR, parameters, T)
# 		plt.plot(tf.reduce_sum(X, axis = 2)[:,0,1], color = "red")
	
# 	ovd_SIR = ovd_sbm_SIR(city_covariates, communities, covariates, observed_index)

# 	prior = tfp.distributions.Gamma(concentration = 2, rate = 2)

# 	for i in range(5):
# 		parameters["Xi"] = prior.sample((np.unique(communities).shape[0], T))
# 		ovd_X, ovd_Y = ovd_simulator(ovd_SIR, parameters, T)
# 		plt.plot(tf.reduce_sum(ovd_X, axis = 2)[:,0,1], color = "orange")

# 	prior = tfp.distributions.Gamma(concentration = 0.1, rate = 0.1)

# 	for i in range(5):
# 		parameters["Xi"] = prior.sample((np.unique(communities).shape[0], T))
# 		ovd_X_noisy, ovd_Y_noisy = ovd_simulator(ovd_SIR, parameters, T)
# 		plt.plot(tf.reduce_sum(ovd_X_noisy, axis = 2)[:,0,1], color = "purple")

# 	print("END")
