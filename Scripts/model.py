import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
def logit(p):

	return tf.math.log(p/(1-p))

def simulation_step_0(pi, sample_size = 1, seed_sim = None):
	"""A single step simulator

	It produces a categorical distribution and simulate from it

	Args:
	pi: Floating point tensor with shape (N,M), probability parameters 
	of the categorical with N population size and M number of compartments
	seed_sim: Floating point tensor with shape (2) representing
	the given seed
	"""

	X = tfp.distributions.Categorical(probs = pi)

	return tf.one_hot(X.sample(sample_size, seed = seed_sim), tf.shape(pi)[1])

def simulation_step(pi, seed_sim = None):
	"""A single step simulator

	It produces a categorical distribution and simulate from it

	Args:
	pi: Floating point tensor with shape (N,M), probability parameters 
	of the categorical with N population size and M number of compartments
	seed_sim: Floating point tensor with shape (2) representing
	the given seed
	"""

	X = tfp.distributions.Categorical(probs = pi)

	return tf.one_hot(X.sample(seed = seed_sim), tf.shape(pi)[-1])

def dynamic_t(x_tm1, k_x_tm1, seed_sim = None):
	"""A single step dynamic

	It produces the next state of the individuals given the current one

	Args:
	x_tm1: Floating point tensor with shape (N,M), one-hot encoding vector
	representing the states of the N individuals
	k_x_tm1: Floating point tensor with shape (N,M,M), collection of N
	stochastic transition matrices representing the probabilities of 
	switching state given the current one
	seed_sim: Floating point tensor with shape (2) representing
	the given seed
	"""

	pi_t = tf.einsum("...np,...npm->...nm", x_tm1, k_x_tm1)

	return simulation_step(pi_t, seed_sim)

def emission_t(x_t, g_t, seed_sim = None):
	"""A single step emission

	It produces the current observations of the individuals given the current state

	Args:
	x_t: Floating point tensor with shape (N,M), one-hot encoding vector
	representing the states of the N individuals
	g_t: Floating point tensor with shape (N,M,M+1), collection of N
	stochastic matrices representing the probabilities of observing an
	individual in a state given the current state
	seed_sim: Floating point tensor with shape (2) representing
	the given seed
	"""

	mu_t = tf.einsum("...np,npm->...nm", x_t, g_t)

	return simulation_step(mu_t, seed_sim)

# @tf.function(jit_compile=True)
def simulator_without_extra_state(ibm, parameters, T, sample_size, seed_sim):
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

		k_x_tm1 = ibm.K_x(parameters, x_tm1)
		x_t = dynamic_t(x_tm1, k_x_tm1, seed_dynamic)

		g_t = ibm.G_t(parameters)
		y_t = emission_t(x_t, g_t, seed_emission)

		return x_t, y_t, seed_carry

	X, Y, _ = tf.scan(body, tf.range(1, T+1), initializer = (x_0, y_0, seed_carry))

	return tf.concat((tf.expand_dims(x_0, axis = 0), X), axis = 0), Y

def simulator_with_extra_state(ibm, parameters, T, sample_size, seed_sim):
	"""An individual based model simulator with extra state

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
	x_0  = simulation_step_0(pi_0, sample_size, seed_sim = seed_0)[0,...]
	y_0 = tf.zeros(tf.shape(x_0), dtype = tf.float32)
	y_0 = tf.concat((y_0, y_0[...,-1:]), axis = -1)
	extra_state = ibm.extra_state

	def body(input, t):

		x_tm1, _, seed_carry, extra_state = input

		seed_dynamic, seed_emission, seed_carry = tfp.random.split_seed( seed_carry, n=3, salt='simulator_step_t')

		k_x_tm1 = ibm.K_x(parameters, x_tm1, extra_state)
		x_t = dynamic_t(x_tm1, k_x_tm1, seed_dynamic)

		g_t = ibm.G_t(parameters, extra_state)
		y_t = emission_t(x_t, g_t, seed_emission)

		extra_state = ibm.extra(parameters, x_t, y_t, extra_state)

		return x_t, y_t, seed_carry, extra_state

	X, Y, _, _ = tf.scan(body, tf.range(1, T+1), initializer = (x_0, y_0, seed_carry, extra_state))

	return tf.concat((tf.expand_dims(x_0, axis = 0), X), axis = 0), Y

def simulator(ibm, parameters, T, sample_size = 1, seed_sim = None):

	if hasattr(ibm, 'extra') and callable(getattr(ibm, 'extra')):
		return simulator_with_extra_state(ibm, parameters, T, sample_size, seed_sim)

	else:
		return simulator_without_extra_state(ibm, parameters, T, sample_size, seed_sim)

# @tf.function(jit_compile=True)
def joint_likelihood(ibm, parameters, X, Y):
	"""An individual based model simulator

	It computes the joint likelihood for the individual based model given the outcome of the simulation.

	Args:
	ibm: an individual based model object, meaning that it require the 
	following methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	parameters: a dictionary with the parameters as keys. This has
	to be build to be compatible with ibm
	X: Floating point tensor with shape (T+1,N,M), representing the states of the N individuals over time
	Y: Floating point tensor with shape (T,N,M), representing the observed states of the N individuals over time
	"""

	T = tf.shape(X)[0]
	
	pi_0 = ibm.pi_0(parameters)
	p_x_0 = tf.einsum("...ni,ni->...n", X[0,...], pi_0)

	def body(input, t):

		x_tm1 = X[t-1,...]
		x_t   = X[t,...]
		y_t   = Y[t-1,...]

		k_x_tm1 = ibm.K_x(parameters, x_tm1)
		p_x_tm1     = tf.einsum("...ni,...nij->...nj", x_tm1,   k_x_tm1)
		p_x_tm1_x_t = tf.einsum("...ni,...ni->...n",   p_x_tm1, x_t)

		g_t = ibm.G_t(parameters)
		g_x_tm1     = tf.einsum("...ni,nij->...nj", x_tm1,   g_t)
		g_x_tm1_x_t = tf.einsum("...ni,...ni->...n",   g_x_tm1, y_t)

		return tf.math.log(p_x_tm1_x_t) + tf.math.log(g_x_tm1_x_t)

	p_X_Y = tf.scan(body, tf.range(1, T), initializer = (tf.math.log(p_x_0)))

	return tf.concat((tf.expand_dims(tf.math.log(p_x_0), axis = 0), p_X_Y), axis = 0)

class basic_SIS():
	"""A Basic SIS 

	We implement a basic SIS model where each individual has some
	individual specific covariates. This qualify as	an individual 
	based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, covariates):
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

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""

		N_float = tf.cast(self.N, dtype = tf.float32)

		infectious_pressure = tf.reduce_sum(x[...,1], axis = -1)/N_float
		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		K_t_n_SI = tf.einsum("...,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = tf.einsum("...,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
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
	

class simba_SIS():
	"""A Basic SIS 

	We implement the SIS considered in the SImulation Based Composite Likelihood paper:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, covariates):
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

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		initial_infection = tf.einsum("nm,m->n", self.covariates, parameters["beta_0"])

		p_0_n_I  = tf.expand_dims(1/(1+tf.math.exp(-initial_infection)), axis = -1)

		return tf.concat((1-p_0_n_I, p_0_n_I), axis = -1)
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""

		N_float = tf.cast(self.N, dtype = tf.float32)

		infectious_pressure = tf.reduce_sum(x[...,1], axis = -1)/N_float
		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		rate_K_t_n_SI = tf.einsum("...,n->...n", infectious_pressure + tf.math.exp(parameters["iota"]), 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SI = 1 - tf.math.exp(-rate_K_t_n_SI)
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		rate_K_t_n_IS = tf.einsum("...,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = 1 - tf.math.exp(-rate_K_t_n_IS)
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
		"""
		The stochastic matrix computing the reporting probabilities.
		Each row refer to a different state, while the columns refer
		to the reporting states, with the first one being unreported

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		G_t_SS_II = tf.expand_dims(tf.linalg.diag(parameters["q"]), axis =0)*tf.ones((self.N, self.M, self.M))
		
		return tf.concat((1 - tf.reduce_sum(G_t_SS_II, axis = -1, keepdims = True), G_t_SS_II), axis = -1)
		

class simba_SEIR():
	"""A Basic SEIR 

	We implement the SEIR considered in the SImulation Based Composite Likelihood paper:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, covariates):
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
		self.M = 4

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		initial_infection = tf.einsum("nm,m->n", self.covariates, parameters["beta_0"])

		p_0_n_I  = tf.expand_dims(1/(1+tf.math.exp(-initial_infection)), axis = -1)

		return tf.concat((1-p_0_n_I, tf.zeros(tf.shape(p_0_n_I)), p_0_n_I, tf.zeros(tf.shape(p_0_n_I))), axis = -1)
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
   
		# W                       = self.covariates
		# beta_lambda, beta_gamma, epsilon, rho = parameters["beta_l"], parameters["beta_g"], tf.math.exp(parameters["iota"]), tf.math.exp(parameters["rho"])

		# lambda__n = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_lambda)))
		# gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,...j->...i", W, beta_gamma)))

		# c_tm1 = tf.reduce_sum( x, axis = -2, keepdims = True ) - x

		# N = tf.cast(tf.shape(W)[0], dtype = tf.float32)

		# rate_SE   = tf.expand_dims(tf.expand_dims(1-tf.exp(-lambda__n*((c_tm1[...,2]/N)+epsilon)), axis =-1), axis =-1)
		# rate_EI   = tf.expand_dims(tf.expand_dims(1 - tf.exp(-rho), axis = -1), axis = -1)*tf.ones(tf.shape(rate_SE))
		# rate_IR   = tf.expand_dims(tf.expand_dims(1-tf.exp(-gamma__n), axis =-1), axis =-1)*tf.ones(tf.shape(rate_SE))

		# K_eta_h__n_r1 = tf.concat((1 - rate_SE, rate_SE, tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE))), axis = -1)
		# K_eta_h__n_r2 = tf.concat((tf.zeros(tf.shape(rate_SE)), 1 - rate_EI, rate_EI, tf.zeros(tf.shape(rate_SE))), axis = -1)    
		# K_eta_h__n_r3 = tf.concat((tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE)), 1 - rate_IR, rate_IR), axis = -1)
		# K_eta_h__n_r4 = tf.concat((tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE)), tf.zeros(tf.shape(rate_SE)), tf.ones(tf.shape(rate_SE))), axis = -1)
		# K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = -2)

		# return K_eta_h__n

		N_float = tf.cast(self.N, dtype = tf.float32)

		infectious_pressure = (tf.reduce_sum(x[...,2], axis = -1, keepdims=True) - x[...,2])/N_float
		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		rate_K_t_n_SE = tf.einsum("...n,n->...n", infectious_pressure + tf.math.exp(parameters["iota"]), 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SE = 1 - tf.math.exp(-rate_K_t_n_SE)
		K_t_n_SE = tf.expand_dims(K_t_n_SE, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SE, K_t_n_SE, tf.zeros(tf.shape(K_t_n_SE)), tf.zeros(tf.shape(K_t_n_SE))), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		rate_K_t_n_EI = tf.ones(tf.shape(K_t_n_SE))*tf.math.exp(parameters["rho"])
		K_t_n_EI = 1 - tf.math.exp(-rate_K_t_n_EI)
		K_t_n_E  = tf.concat((tf.zeros(tf.shape(K_t_n_EI)), 1 - K_t_n_EI, K_t_n_EI, tf.zeros(tf.shape(K_t_n_EI))), axis = -1)
		K_t_n_E  = tf.expand_dims(K_t_n_E, axis = -2)

		rate_K_t_n_IR = tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IR = 1 - tf.math.exp(-rate_K_t_n_IR)
		K_t_n_IR = tf.expand_dims(K_t_n_IR, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), tf.zeros(tf.shape(K_t_n_IR)), 1 - K_t_n_IR, K_t_n_IR), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_IR)), tf.zeros(tf.shape(K_t_n_IR)), tf.zeros(tf.shape(K_t_n_IR)), tf.ones(tf.shape(K_t_n_IR))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_E, K_t_n_I, K_t_n_R), axis  = -2)

	def G_t(self, parameters):
		"""
		The stochastic matrix computing the reporting probabilities.
		Each row refer to a different state, while the columns refer
		to the reporting states, with the first one being unreported

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		G_t_SS_II = tf.expand_dims(tf.linalg.diag(parameters["q"]), axis =0)*tf.ones((self.N, self.M, self.M))
		
		return tf.concat((1 - tf.reduce_sum(G_t_SS_II, axis = -1, keepdims = True), G_t_SS_II), axis = -1)

class spatial_SIS():
	"""A Spatial SIS 

	We implement a spatial SIS model where each individual has a
	location and some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, locations, covariates):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.locations  = locations
		self.covariates = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 2

	def spatial_matrix(self, parameters):
		"""

		This build the interaction matrix using a Gaussian kernel. The rows
		of the matrix are normalized to sum to N.

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		d_1 = tf.math.pow(self.locations[:,0:1] - tf.transpose(self.locations[:,0:1]), 2)
		d_2 = tf.math.pow(self.locations[:,1:2] - tf.transpose(self.locations[:,1:2]), 2)
		d = tf.math.sqrt(d_1 + d_2)

		spatial_matrix = tf.math.exp(parameters["log_chi"])*tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_phi"]/2)).prob(d)
		mask           = 1-tf.eye(self.N)

		# remove the diagonal (individuals cannot infect themselves)
		spatial_matrix = spatial_matrix*mask

		return spatial_matrix

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""

		N_float = tf.cast(self.N, dtype = tf.float32)

		infectious_pressure = tf.einsum("...i,ji->...j", x[...,1], self.spatial_matrix(parameters) )/N_float
		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		K_t_n_SI =  1 - tf.math.exp(-tf.einsum("...n,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility))) - tf.math.exp(parameters["log_epsilon"]))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = 1 - tf.math.exp(-tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability))))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
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

class old_spatial_SIS():
	"""A Spatial SIS 

	We implement a spatial SIS model where each individual has a
	location and some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, locations, covariates):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.locations  = locations
		self.covariates = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 2

	def spatial_matrix(self, parameters):
		"""

		This build the interaction matrix using a Gaussian kernel. The rows
		of the matrix are normalized to sum to N.

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		d_1 = tf.math.pow(self.locations[:,0:1] - tf.transpose(self.locations[:,0:1]), 2)
		d_2 = tf.math.pow(self.locations[:,1:2] - tf.transpose(self.locations[:,1:2]), 2)
		d = tf.math.sqrt(d_1 + d_2)

		spatial_matrix = tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_phi"])).prob(d)
		mask           = 1-tf.eye(self.N)

		# remove the diagonal (individuals cannot infect themselves)
		spatial_matrix = spatial_matrix*mask

		# rows_normaliz = tf.reduce_sum(tf.linalg.band_part(new_matrix, 0, -1), axis =1, keepdims=True)
		# cols_normaliz = tf.reduce_sum(tf.linalg.band_part(new_matrix, 0, -1), axis =0, keepdims=True)

		# den_rows_normaliz = rows_normaliz + tf.expand_dims(tf.one_hot(tf.shape(rows_normaliz)[0]-1, tf.shape(rows_normaliz)[0]), axis = 1)

		# self_normalized = tf.linalg.band_part(new_matrix, 0, -1)/(rows_normaliz + tf.transpose(cols_normaliz)/den_rows_normaliz)
		# self_normalized = self_normalized + tf.transpose(self_normalized)

		return spatial_matrix/tf.reduce_sum(spatial_matrix, axis = 1, keepdims=True)

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""

		infectious_pressure = tf.einsum("...i,ji->...j", x[...,1], self.spatial_matrix(parameters) )
		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		K_t_n_SI = tf.einsum("...n,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
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
	

class spatial_SIS_env():
	"""A Spatial SIS 

	We implement a spatial SIS model where each individual has a
	location and some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, locations, covariates):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.locations  = locations
		self.covariates = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 2

	def spatial_matrix(self, parameters):
		"""

		This build the interaction matrix using a Gaussian kernel. The rows
		of the matrix are normalized to sum to N.

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		d_1 = tf.math.pow(self.locations[:,0:1] - tf.transpose(self.locations[:,0:1]), 2)
		d_2 = tf.math.pow(self.locations[:,1:2] - tf.transpose(self.locations[:,1:2]), 2)
		d = tf.math.sqrt(d_1 + d_2)

		spatial_matrix = tfp.distributions.Normal(loc = 0, scale = tf.math.exp(parameters["log_phi"])).prob(d)
		mask           = 1-tf.eye(self.N)

		# remove the diagonal (individuals cannot infect themselves)
		spatial_matrix = spatial_matrix*mask

		# rows_normaliz = tf.reduce_sum(tf.linalg.band_part(new_matrix, 0, -1), axis =1, keepdims=True)
		# cols_normaliz = tf.reduce_sum(tf.linalg.band_part(new_matrix, 0, -1), axis =0, keepdims=True)

		# den_rows_normaliz = rows_normaliz + tf.expand_dims(tf.one_hot(tf.shape(rows_normaliz)[0]-1, tf.shape(rows_normaliz)[0]), axis = 1)

		# self_normalized = tf.linalg.band_part(new_matrix, 0, -1)/(rows_normaliz + tf.transpose(cols_normaliz)/den_rows_normaliz)
		# self_normalized = self_normalized + tf.transpose(self_normalized)

		return spatial_matrix

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""

		infectious_pressure = tf.einsum("...i,ji->...j", x[...,1], self.spatial_matrix(parameters) ) + tf.math.exp(parameters["log_epsilon"])
		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		K_t_n_SI = 1 - tf.math.exp(-tf.einsum("...n,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility))))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
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

class sbm_SIS():
	"""A stochastic block model SIS 

	We implement a stochastic block model where each individual has a
	community and some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, communities, covariates):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.communities = communities
		self.covariates  = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 2

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
	
	def B_matrix(self, parameters):
		
		Ncommunities = tf.shape(self.communities)[1]
		beta = tf.math.exp(parameters["log_graph"])

		B = tf.math.exp(-beta)*tf.eye(Ncommunities)
		sovra_diag       = tf.concat((tf.concat((tf.zeros((Ncommunities-1, 1)), tf.eye(Ncommunities-1)), axis = 1), tf.zeros((1, Ncommunities))), axis = 0)
		sovra_sovra_diag = tf.concat((tf.concat((tf.zeros((Ncommunities-1, 1)), sovra_diag[:-1,:-1]), axis = 1), tf.zeros((1, Ncommunities))), axis = 0)
		sovra_sovra_sovra_diag = tf.concat((tf.concat((tf.zeros((Ncommunities-1, 1)), sovra_sovra_diag[:-1,:-1]), axis = 1), tf.zeros((1, Ncommunities))), axis = 0)
		B = tf.experimental.numpy.triu(B + tf.math.exp(-10*beta)*sovra_diag + tf.math.exp(-20*beta)*sovra_sovra_diag + tf.math.exp(-30*beta)*sovra_sovra_sovra_diag)
		B = B + tf.transpose(tf.experimental.numpy.triu(B, k = 1))

		return B
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
		I_c      = tf.einsum("...c,c->...c", tf.einsum("...n,nc->...c", x[...,1], self.communities), 1/tf.reduce_sum(self.communities, axis = 0))
		prob_I_c = tf.einsum("ij,...j->...i", self.B_matrix(parameters), I_c)
		infectious_pressure = tf.einsum("nc,...c->...n", self.communities, prob_I_c) + tf.math.exp(parameters["log_epsilon"])

		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		K_t_n_SI = tf.einsum("...n,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
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
	

class sbm_SIS_finite():
	"""A stochastic block model SIS 

	We implement a stochastic block model where each individual has a
	community and some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, communities, covariates):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.communities = communities
		self.covariates  = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 2

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
		I_c      = tf.einsum("...c,c->...c", tf.einsum("...n,nc->...c", x[...,1], self.communities), 1/tf.reduce_sum(self.communities, axis = 0))
		prob_I_c = tf.einsum("ij,...j->...i", parameters["B_matrix"], I_c)
		infectious_pressure = tf.einsum("nc,...c->...n", self.communities, prob_I_c) + tf.math.exp(parameters["log_epsilon"])

		susceptibility      = tf.einsum("ni,i->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("ni,i->n", self.covariates, parameters["beta_g"])

		K_t_n_SI = tf.einsum("...n,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((K_t_n_IS, 1 - K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I), axis  = -2)
    
	def G_t(self, parameters):
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
	

class network_SIR():
	"""An SIR model on a given graph 

	We implement an individual-based model where each individual is connected to some other individuals 
	and it presents some individual specific covariates. This qualify as
	an individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, connection_matrix, covariates):
		super().__init__()
		"""Construct the spatial SIS

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,3)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.connection_matrix = connection_matrix
		self.covariates  = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 3

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		return parameters["prior_infection"]*tf.ones((self.N, self.M))
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
		infectious_pressure      = tf.einsum("...j,j->...j", tf.einsum("...j,ij->...i", x[...,1], self.connection_matrix), 1/tf.reduce_sum(self.connection_matrix, axis = 1))

		susceptibility      = tf.einsum("nm,m->n", self.covariates, parameters["beta_l"])
		recoverability      = tf.einsum("nm,m->n", self.covariates, parameters["beta_g"])

		K_t_n_SI = tf.einsum("...n,n->...n", infectious_pressure, 1/(1+tf.math.exp(-susceptibility)))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI, tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		K_t_n_IS = tf.einsum("...n,n->...n", tf.ones(tf.shape(infectious_pressure)), 1/(1+tf.math.exp(-recoverability)))
		K_t_n_IS = tf.expand_dims(K_t_n_IS, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), 1 - K_t_n_IS, K_t_n_IS), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		  
		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.ones(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_R), axis  = -2)
    
	def G_t(self, parameters):
		"""
		The stochastic matrix computing the reporting probabilities.
		Each row refer to a different state, while the columns refer
		to the reporting states, with the first one being unreported

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		G_t_reported = tf.expand_dims(tf.linalg.diag(tf.math.sigmoid(parameters["logit_prob_testing"])), axis =0)*tf.ones((self.N, self.M, self.M))

		return tf.concat((1 - tf.reduce_sum(G_t_reported, axis = -1, keepdims=True), G_t_reported), axis = -1)
	
# if __name__ == "__main__":
# 	import time

# 	N_pop = 1000
# 	covariates = tf.convert_to_tensor(np.load("CAL/Data/SpatialInference/Input/covariates.npy"), dtype = tf.float32)[:N_pop,:]
# 	locations  = tf.convert_to_tensor(np.load("CAL/Data/SpatialInference/Input/locations.npy"), dtype = tf.float32)[:N_pop,:]

# 	N = tf.shape(covariates)[0]

# 	parameters = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
# 		"beta_l":tf.convert_to_tensor([-1.0, +2.0], dtype = tf.float32),
# 		"beta_g":tf.convert_to_tensor([-1.0, -1.0], dtype = tf.float32),
# 		"logit_sensitivity":logit(
# 			tf.convert_to_tensor([0.9], dtype = tf.float32)),
# 		"logit_specificity":logit(
# 			tf.convert_to_tensor([0.95], dtype = tf.float32)),
# 		"logit_prob_testing":logit(
# 			tf.convert_to_tensor([0.1, 0.2], dtype = tf.float32)),}

# 	SIS = basic_SIS(covariates)

# 	T    = 200
# 	start = time.time()
# 	X, Y = simulator(SIS, parameters, T)
# 	print(time.time()-start)