from model import *

def FM_simulation_step_0(pi, seed_sim = None):
	"""A single step simulator

	It produces a categorical distribution and simulate from it

	Args:
	pi: Floating point tensor with shape (N,M), probability parameters 
	of the categorical with N population size and M number of compartments
	seed_sim: Floating point tensor with shape (2) representing
	the given seed
	"""

	X = tfp.distributions.Categorical(probs = pi)

	return tf.one_hot(X.sample(seed = seed_sim), tf.shape(pi)[1])

# @tf.function(jit_compile=True)
def FM_simulator_without_extra_state(ibm, parameters, T, seed_sim):
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
	x_0  = FM_simulation_step_0(pi_0, seed_sim = seed_0)
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

def FM_simulator_with_extra_state(ibm, parameters, T, seed_sim):
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
	x_0  = FM_simulation_step_0(pi_0, seed_sim = seed_0)[0,...]
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

def FM_simulator(ibm, parameters, T, seed_sim = None):

	if hasattr(ibm, 'extra') and callable(getattr(ibm, 'extra')):
		return FM_simulator_with_extra_state(ibm, parameters, T, seed_sim)

	else:
		return FM_simulator_without_extra_state(ibm, parameters, T, seed_sim)

class sparse_FM_SINR():
	"""An SIQR model for foot and mouth data 

	We implement an individual-based model for the foot and mouth data. 
	Each farm is located spatially in the UK, with covariates given by 
	nr of cattles and nr of sheeps. The spatial interaction decreases
	in space according to a Cauchy-type kernel. This qualify as an 
	individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	NB: it does not support parallel simulations
	"""

	def __init__(self, values, indeces, covariates):
		super().__init__()
		"""Construct the spatial SIQR

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.values     = values
		self.indeces    = indeces
		self.covariates = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 4
		self.parallel_samples = False

	# def pi_0(self, parameters):
	# 	"""
	# 	The initial probability of being infected

	# 	Args:
	# 		parameters: a dictionary with the parameters as keys
	# 	"""

	# 	pi_0_S = tf.expand_dims(1-tf.sigmoid(parameters["logit_prior_infection"]), axis = -1)

	# 	return tf.concat((pi_0_S, 1-pi_0_S, tf.zeros(tf.shape(pi_0_S)), tf.zeros(tf.shape(pi_0_S)) ), axis = -1)

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		infectivity    = tf.exp(parameters["log_zeta"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))
		x_infectivity = tf.einsum("n,n->...n", tf.ones(self.N, dtype = tf.float32), infectivity)/tf.cast(self.N, tf.float32)

		susceptibility = tf.exp(parameters["log_xi"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))

		infectious_pressure = susceptibility*self.spatial_infectious_pressure(tf.exp(parameters["log_psi"]), x_infectivity)

		pi_0_S = tf.exp(-tf.exp(parameters["log_tau"])*tf.exp(parameters["log_delta"])*infectious_pressure/tf.cast(self.N, tf.float32) - tf.exp(parameters["log_epsilon"]))

		return tf.stack((pi_0_S, 1-pi_0_S, tf.zeros(tf.shape(pi_0_S)), tf.zeros(tf.shape(pi_0_S)) ), axis = -1)
	
	def spatial_infectious_pressure(self, psi, infected):

		infection_pressure_ij = tf.gather(infected, self.indeces[:,1], axis = 0)*(psi/(tf.math.pow(self.values, 2) + tf.math.pow(psi, 2)))

		return tf.tensor_scatter_nd_add(tf.zeros(tf.shape(infected)), self.indeces[:,0:1], infection_pressure_ij)
    
	def K_x(self, parameters, x):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
		infectivity    = tf.exp(parameters["log_zeta"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))
		x_infectivity = tf.einsum("...n,n->...n", x[...,1], infectivity)

		susceptibility = tf.exp(parameters["log_xi"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))

		infectious_pressure = susceptibility*self.spatial_infectious_pressure(tf.exp(parameters["log_psi"]), x_infectivity)

		K_t_n_SI = 1 - tf.exp(-tf.exp(parameters["log_delta"])*infectious_pressure/tf.cast(self.N, tf.float32) - tf.exp(parameters["log_epsilon"]))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI, tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		quarantine_rate = tf.ones(tf.shape(x[...,1]))*tf.exp(parameters["log_gamma"])

		K_t_n_IN = 1 - tf.exp(-quarantine_rate)
		K_t_n_IN = tf.expand_dims(K_t_n_IN, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), 1 - K_t_n_IN, K_t_n_IN, tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		removal_prob = tf.ones(tf.shape(x[...,1]))

		K_t_n_NR = removal_prob
		K_t_n_NR = tf.expand_dims(K_t_n_NR, axis = -1)  
		K_t_n_N  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), 1 - K_t_n_NR, K_t_n_NR), axis = -1)
		K_t_n_N  = tf.expand_dims(K_t_n_N, axis = -2)

		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.ones(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_N, K_t_n_R), axis  = -2)
    
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
	

class sparse_FM_SIQR():
	"""An SIQR model for foot and mouth data 

	We implement an individual-based model for the foot and mouth data. 
	Each farm is located spatially in the UK, with covariates given by 
	nr of cattles and nr of sheeps. The spatial interaction decreases
	in space according to a Cauchy-type kernel. This qualify as an 
	individual based model object, meaning that it has the following 
	methods:
		- pi_0(parameters): to compute the initial distribution
		- K_x(parameters, x): to comupte the transition kernel
		- G_t(parameters): to compute the emission matrix
	"""

	def __init__(self, values, indeces, covariates):
		super().__init__()
		"""Construct the spatial SIQR

		This creates some quantities that do not change

		Args:
			locations: Floating point tensor with shape (N,2)
			covariates: Floating point tensor with shape (N,C), with C number of covariates 
			representing which parameters are know which are not. The default assume 
			everything is unknown.
		"""

		self.values     = values
		self.indeces    = indeces
		self.covariates = covariates

		self.N = tf.shape(covariates)[0]
		self.M = 4

		self.extra_state = tf.zeros(self.N)
		self.parallel_samples = False

	def pi_0(self, parameters):
		"""
		The initial probability of being infected

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		pi_0_S = tf.expand_dims(1-tf.sigmoid(parameters["logit_prior_infection"]), axis = -1)

		return tf.concat((pi_0_S, 1-pi_0_S, tf.zeros(tf.shape(pi_0_S)), tf.zeros(tf.shape(pi_0_S)) ), axis = -1)

	def spatial_infectious_pressure(self, psi, infected):

		infection_pressure_ij = tf.gather(infected, self.indeces[:,1], axis = 0)*(psi/(tf.math.pow(self.values, 2) + tf.math.pow(psi, 2)))

		return tf.tensor_scatter_nd_add(tf.zeros(tf.shape(infected)), self.indeces[:,0:1], infection_pressure_ij)
    
	def K_x(self, parameters, x, extra_state):
		"""
		The stochastic transition matrix computing the probabilities
		of moving across states given a population state x.

		Args:
			parameters: a dictionary with the parameters as keys
			x: Floating point tensor with shape (N,2)
		"""
		detected = extra_state

		infectivity    = tf.exp(parameters["log_zeta"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))
		x_infectivity = tf.einsum("...n,n->...n", x[...,1], infectivity)

		susceptibility = tf.exp(parameters["log_xi"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))

		infectious_pressure = susceptibility*self.spatial_infectious_pressure(tf.exp(parameters["log_psi"]), x_infectivity)

		K_t_n_SI = 1 - tf.exp(-tf.exp(parameters["log_delta"])*infectious_pressure/tf.cast(self.N, tf.float32) - tf.exp(parameters["log_epsilon"]))
		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI, tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

		quarantine_rate = tf.einsum("...n,...n->...n", detected, tf.ones(tf.shape(x[...,1])))*tf.exp(parameters["log_q_rate"])
		stay_infected   = tf.einsum("...n,...n->...n", 1 - detected, tf.ones(tf.shape(x[...,1])))

		K_t_n_IQ = 1 - tf.exp(-quarantine_rate)
		K_t_n_IQ = tf.expand_dims(K_t_n_IQ, axis = -1)  
		K_t_n_II = tf.expand_dims(stay_infected, axis = -1)  
		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), K_t_n_II, (1-K_t_n_II)*K_t_n_IQ, (1-K_t_n_II)*(1 - K_t_n_IQ)), axis = -1)
		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

		removal_rate = tf.ones(tf.shape(x[...,1]))*tf.exp(parameters["log_r_rate"])

		K_t_n_QR = 1 - tf.exp(-removal_rate)
		K_t_n_QR = tf.expand_dims(K_t_n_QR, axis = -1)  
		K_t_n_Q  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), 1 - K_t_n_QR, K_t_n_QR), axis = -1)
		K_t_n_Q  = tf.expand_dims(K_t_n_Q, axis = -2)

		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.ones(tf.shape(K_t_n_SI))), axis = -1)
		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_Q, K_t_n_R), axis  = -2)
    
	def G_t(self, parameters, extra_state):
		"""
		The stochastic matrix computing the reporting probabilities.
		Each row refer to a different state, while the columns refer
		to the reporting states, with the first one being unreported

		Args:
			parameters: a dictionary with the parameters as keys
		"""

		G_t_reported = tf.expand_dims(tf.linalg.diag(tf.math.sigmoid(parameters["logit_prob_testing"])), axis =0)*tf.ones((self.N, self.M, self.M))

		return tf.concat((1 - tf.reduce_sum(G_t_reported, axis = -1, keepdims=True), G_t_reported), axis = -1)
    	
	def extra(self, parameters, x_t, y_t, extra_state):

		return y_t[...,2]

# if __name__ == "__main__":

# 	import time	
# 	import sys
# 	sys.path.append('Scripts/')
# 	from model import *
# 	from CAL import *

# 	indexes = tf.convert_to_tensor(np.load("Data/FM/indexes_FM_cumbria_radius50.npy"), dtype = tf.int64)
# 	values  = tf.convert_to_tensor(np.load("Data/FM/values_FM_cumbria_radius50.npy"), dtype = tf.float32)
# 	covariates = tf.convert_to_tensor(np.load("Data/FM/covariates_FM_cumbria.npy"), dtype = tf.float32)

# 	Y = tf.convert_to_tensor(np.load("Data/FM/Y_FM_cumbria.npy"), dtype = tf.float32)[:10,...]
# 	# Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria_SIQR.npy"), dtype = tf.float32)
# 	time_before_infection = tf.cast(tf.shape(Y)[0], tf.float32) - tf.reduce_sum(tf.math.cumsum( Y[...,2], axis = 0), axis = 0)

# 	parameters = {#"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 		"log_tau":tf.convert_to_tensor([np.log(40)], dtype = tf.float32),
# 	        "log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.Variable(tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32)),
# 		"log_gamma":tf.convert_to_tensor([np.log(0.5)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}
	
# 	FM_ibm = sparse_FM_SINR(values, indexes, covariates)
# 	# FM_ibm = sparse_FM_SIQR(values, indexes, covariates)

# 	T    = 100
# 	start = time.time()
# 	X, Y = FM_simulator(FM_ibm, parameters, T)
# 	print(time.time()-start)

# 	start = time.time()
# 	loss = CAL_compiled(FM_ibm, parameters, Y)
# 	print(time.time()-start)

# if __name__ == "__main__":

# 	locations  = tf.convert_to_tensor(np.load("CAL/Data/FM/locations_FM_cumbria.npy"), dtype = tf.float32)
# 	covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/covariates_FM_cumbria.npy"), dtype = tf.float32)
# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria.npy"), dtype = tf.float32)
# 	time_before_infection = tf.cast(tf.shape(Y)[0], tf.float32) - tf.reduce_sum(tf.math.cumsum( Y[...,3], axis = 0), axis = 0)

# 	batch_size = 1000

# 	parameters = {"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 	        "log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.0001)], dtype = tf.float32),
# 		"log_gamma":tf.convert_to_tensor([np.log(0.5)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}

# 	FM_ibm = FM_SINR(locations, covariates, batch_size)

# 	T    = 100
	
# 	X, Y = simulator(FM_ibm, parameters, T)


# if __name__ == "__main__":

# 	indeces  = tf.convert_to_tensor(np.load("CAL/Data/FM/indexes_FM_cumbria.npy"), dtype = tf.int64)
# 	values  = tf.convert_to_tensor(np.load("CAL/Data/FM/values_FM_cumbria.npy"), dtype = tf.float32)
# 	covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/covariates_FM_cumbria.npy"), dtype = tf.float32)
# 	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/Y_FM_cumbria.npy"), dtype = tf.float32)
# 	time_before_infection = tf.cast(tf.shape(Y)[0], tf.float32) - tf.reduce_sum(tf.math.cumsum( Y[...,3], axis = 0), axis = 0)

# 	batch_size = 1000

# 	parameters = {"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5)),
# 	        "log_delta":tf.convert_to_tensor([np.log(0.001)], dtype = tf.float32),
# 		"log_zeta":tf.convert_to_tensor([np.log(100)], dtype = tf.float32),
# 		"log_xi":tf.convert_to_tensor([np.log(10.0)], dtype = tf.float32),
# 		"log_chi":tf.convert_to_tensor([np.log(0.6)], dtype = tf.float32),
# 		"log_psi":tf.convert_to_tensor([np.log(2.0)], dtype = tf.float32),
# 		"log_epsilon":tf.convert_to_tensor([np.log(0.0001)], dtype = tf.float32),
# 		"log_gamma":tf.convert_to_tensor([np.log(0.5)], dtype = tf.float32),
# 		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}

# 	FM_ibm = sparse_FM_SINR(values, indeces, covariates)

# 	T    = 100
	
# 	X, Y = sparse_simulator(FM_ibm, parameters, T)

# class FM_SIQR():
# 	"""An SIQR model for foot and mouth data 

# 	We implement an individual-based model for the foot and mouth data. 
# 	Each farm is located spatially in the UK, with covariates given by 
# 	nr of cattles and nr of sheeps. The spatial interaction decreases
# 	in space according to a Cauchy-type kernel. This qualify as an 
# 	individual based model object, meaning that it has the following 
# 	methods:
# 		- pi_0(parameters): to compute the initial distribution
# 		- K_x(parameters, x): to comupte the transition kernel
# 		- G_t(parameters): to compute the emission matrix
# 	"""

# 	def __init__(self, locations, covariates, batch_size = 1000):
# 		super().__init__()
# 		"""Construct the spatial SIQR

# 		This creates some quantities that do not change

# 		Args:
# 			locations: Floating point tensor with shape (N,2)
# 			covariates: Floating point tensor with shape (N,C), with C number of covariates 
# 			representing which parameters are know which are not. The default assume 
# 			everything is unknown.
# 		"""

# 		self.locations  = locations
# 		self.covariates = covariates

# 		self.batch_size = batch_size

# 		self.N = tf.shape(covariates)[0]
# 		self.M = 4

# 		self.detected = tf.zeros(self.N)

# 	def pi_0(self, parameters):
# 		"""
# 		The initial probability of being infected

# 		Args:
# 			parameters: a dictionary with the parameters as keys
# 		"""

# 		pi_0_S = tf.expand_dims(1-tf.sigmoid(parameters["logit_prior_infection"]), axis = -1)

# 		return tf.concat((pi_0_S, 1-pi_0_S, tf.zeros(tf.shape(pi_0_S)), tf.zeros(tf.shape(pi_0_S)) ), axis = -1)

# 	def infectious_pressure(self, parameters, x):

# 		final_size = tf.cast(self.N%self.batch_size, tf.int32)
# 		iterations = tf.cast((self.N - final_size)/self.batch_size, tf.int32)

# 		infectivity    = tf.exp(parameters["log_zeta"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))
# 		x_infectivity = tf.einsum("...n,n->...n", x[...,1], infectivity)

# 		susceptibility = tf.exp(parameters["log_xi"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))

# 		def cond(i, cum_infectious_pressure):
# 			return i<iterations

# 		def body(i, cum_infectious_pressure):

# 			batch_distances_2 = tf.math.pow((self.locations[:,0:1] - tf.transpose(self.locations[i*self.batch_size:(i+1)*self.batch_size,0:1]))/tf.cast(1000, dtype = tf.float32), 2) + tf.math.pow((self.locations[:,1:2] - tf.transpose(self.locations[i*self.batch_size:(i+1)*self.batch_size,1:2]))/tf.cast(1000, dtype = tf.float32), 2)
# 			kernel_batch      = tf.exp(parameters["log_psi"])/(batch_distances_2 + tf.exp(2*parameters["log_psi"]))

# 			infectious_pressure_batch = tf.einsum("nb,...b->...n", kernel_batch, x_infectivity[...,i*self.batch_size:(i+1)*self.batch_size])

# 			return i+1, cum_infectious_pressure + infectious_pressure_batch

# 		_, cum_infectious_pressure = tf.while_loop(cond, body, loop_vars = (tf.constant(0), tf.zeros(tf.shape(x[...,1]))))

# 		batch_distances_2 = tf.math.pow((self.locations[:,0:1] - tf.transpose(self.locations[iterations*self.batch_size:,0:1]))/tf.cast(1000, dtype = tf.float32), 2) + tf.math.pow((self.locations[:,1:2] - tf.transpose(self.locations[iterations*self.batch_size:,1:2]))/tf.cast(1000, dtype = tf.float32), 2)
# 		kernel_batch      = tf.exp(parameters["log_psi"])/(batch_distances_2 + tf.exp(2*parameters["log_psi"]))

# 		infectious_pressure_batch = tf.einsum("nb,...b->...n", kernel_batch, x_infectivity[...,iterations*self.batch_size:])

# 		return tf.einsum("n,...n->...n", susceptibility, cum_infectious_pressure + infectious_pressure_batch)
    
# 	def K_x(self, parameters, x):
# 		"""
# 		The stochastic transition matrix computing the probabilities
# 		of moving across states given a population state x.

# 		Args:
# 			parameters: a dictionary with the parameters as keys
# 			x: Floating point tensor with shape (N,2)
# 		"""
# 		infectious_pressure = self.infectious_pressure(parameters, x)

# 		K_t_n_SI = 1 - tf.exp(-tf.exp(parameters["log_delta"])*infectious_pressure)
# 		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
# 		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI, tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
# 		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

# 		quarantine_rate = tf.einsum("...n,...n->...n", self.detected, tf.ones(tf.shape(x[...,1])))*tf.exp(parameters["log_q_rate"])
# 		stay_infected   = tf.einsum("...n,...n->...n", 1 - self.detected, tf.ones(tf.shape(x[...,1])))

# 		K_t_n_IQ = 1 - tf.exp(-quarantine_rate)
# 		K_t_n_IQ = tf.expand_dims(K_t_n_IQ, axis = -1)  
# 		K_t_n_II = tf.expand_dims(stay_infected, axis = -1)  
# 		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), K_t_n_II, (1-K_t_n_II)*K_t_n_IQ, (1-K_t_n_II)*(1 - K_t_n_IQ)), axis = -1)
# 		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

# 		removal_rate = tf.ones(tf.shape(x[...,1]))*tf.exp(parameters["log_r_rate"])

# 		K_t_n_QR = 1 - tf.exp(-removal_rate)
# 		K_t_n_QR = tf.expand_dims(K_t_n_QR, axis = -1)  
# 		K_t_n_Q  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), 1 - K_t_n_QR, K_t_n_QR), axis = -1)
# 		K_t_n_Q  = tf.expand_dims(K_t_n_Q, axis = -2)

# 		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.ones(tf.shape(K_t_n_SI))), axis = -1)
# 		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

# 		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_Q, K_t_n_R), axis  = -2)
    
# 	def G_t(self, parameters):
# 		"""
# 		The stochastic matrix computing the reporting probabilities.
# 		Each row refer to a different state, while the columns refer
# 		to the reporting states, with the first one being unreported

# 		Args:
# 			parameters: a dictionary with the parameters as keys
# 		"""

# 		G_t_reported = tf.expand_dims(tf.linalg.diag(tf.math.sigmoid(parameters["logit_prob_testing"])), axis =0)*tf.ones((self.N, self.M, self.M))

# 		return tf.concat((1 - tf.reduce_sum(G_t_reported, axis = -1, keepdims=True), G_t_reported), axis = -1)
    	
# 	def extra(self, parameters, x_t, y_t):

# 		self.detected = y_t[...,2]

# class FM_SINR():
# 	"""An SIQR model for foot and mouth data 

# 	We implement an individual-based model for the foot and mouth data. 
# 	Each farm is located spatially in the UK, with covariates given by 
# 	nr of cattles and nr of sheeps. The spatial interaction decreases
# 	in space according to a Cauchy-type kernel. This qualify as an 
# 	individual based model object, meaning that it has the following 
# 	methods:
# 		- pi_0(parameters): to compute the initial distribution
# 		- K_x(parameters, x): to comupte the transition kernel
# 		- G_t(parameters): to compute the emission matrix
# 	"""

# 	def __init__(self, locations, covariates, batch_size = 1000):
# 		super().__init__()
# 		"""Construct the spatial SIQR

# 		This creates some quantities that do not change

# 		Args:
# 			locations: Floating point tensor with shape (N,2)
# 			covariates: Floating point tensor with shape (N,C), with C number of covariates 
# 			representing which parameters are know which are not. The default assume 
# 			everything is unknown.
# 		"""

# 		self.locations  = locations
# 		self.covariates = covariates

# 		self.batch_size = batch_size

# 		self.N = tf.shape(covariates)[0]
# 		self.M = 4

# 		self.detected = tf.zeros(self.N)

# 	def pi_0(self, parameters):
# 		"""
# 		The initial probability of being infected

# 		Args:
# 			parameters: a dictionary with the parameters as keys
# 		"""

# 		pi_0_S = tf.expand_dims(1-tf.sigmoid(parameters["logit_prior_infection"]), axis = -1)

# 		return tf.concat((pi_0_S, 1-pi_0_S, tf.zeros(tf.shape(pi_0_S)), tf.zeros(tf.shape(pi_0_S)) ), axis = -1)

# 	def infectious_pressure(self, parameters, x):

# 		final_size = tf.cast(self.N%self.batch_size, tf.int32)
# 		iterations = tf.cast((self.N - final_size)/self.batch_size, tf.int32)

# 		infectivity    = tf.exp(parameters["log_zeta"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))
# 		x_infectivity = tf.einsum("...n,n->...n", x[...,1], infectivity)

# 		susceptibility = tf.exp(parameters["log_xi"])*tf.math.pow(self.covariates[:,0], tf.exp(parameters["log_chi"])) + tf.math.pow(self.covariates[:,1], tf.exp(parameters["log_chi"]))

# 		# def cond(i, cum_infectious_pressure):
# 		# 	return i<iterations

# 		# def body(i, cum_infectious_pressure):

# 		# 	batch_distances_2 = tf.math.pow((self.locations[:,0:1] - tf.transpose(self.locations[i*self.batch_size:(i+1)*self.batch_size,0:1]))/tf.cast(1000, dtype = tf.float32), 2) + tf.math.pow((self.locations[:,1:2] - tf.transpose(self.locations[i*self.batch_size:(i+1)*self.batch_size,1:2]))/tf.cast(1000, dtype = tf.float32), 2)
# 		# 	kernel_batch      = tf.exp(parameters["log_psi"])/(batch_distances_2 + tf.exp(2*parameters["log_psi"]))

# 		# 	infectious_pressure_batch = tf.einsum("nb,...b->...n", kernel_batch, x_infectivity[...,i*self.batch_size:(i+1)*self.batch_size])

# 		# 	return i+1, cum_infectious_pressure + infectious_pressure_batch

# 		# _, cum_infectious_pressure = tf.while_loop(cond, body, loop_vars = (tf.constant(0), tf.zeros(tf.shape(x[...,1]))))

# 		cum_infectious_pressure = tf.zeros(tf.shape(x[...,1]))
# 		for i in range(iterations):
# 			batch_distances_2 = tf.math.pow((self.locations[:,0:1] - tf.transpose(self.locations[i*self.batch_size:(i+1)*self.batch_size,0:1]))/tf.cast(1000, dtype = tf.float32), 2) + tf.math.pow((self.locations[:,1:2] - tf.transpose(self.locations[i*self.batch_size:(i+1)*self.batch_size,1:2]))/tf.cast(1000, dtype = tf.float32), 2)
# 			kernel_batch      = tf.exp(parameters["log_psi"])/(batch_distances_2 + tf.exp(2*parameters["log_psi"]))

# 			infectious_pressure_batch = tf.einsum("nb,...b->...n", kernel_batch, x_infectivity[...,i*self.batch_size:(i+1)*self.batch_size])

# 			cum_infectious_pressure = cum_infectious_pressure + infectious_pressure_batch

# 		batch_distances_2 = tf.math.pow((self.locations[:,0:1] - tf.transpose(self.locations[iterations*self.batch_size:,0:1]))/tf.cast(1000, dtype = tf.float32), 2) + tf.math.pow((self.locations[:,1:2] - tf.transpose(self.locations[iterations*self.batch_size:,1:2]))/tf.cast(1000, dtype = tf.float32), 2)
# 		kernel_batch      = tf.exp(parameters["log_psi"])/(batch_distances_2 + tf.exp(2*parameters["log_psi"]))

# 		infectious_pressure_batch = tf.einsum("nb,...b->...n", kernel_batch, x_infectivity[...,iterations*self.batch_size:])

# 		return tf.einsum("n,...n->...n", susceptibility, cum_infectious_pressure + infectious_pressure_batch)
    
# 	def K_x(self, parameters, x):
# 		"""
# 		The stochastic transition matrix computing the probabilities
# 		of moving across states given a population state x.

# 		Args:
# 			parameters: a dictionary with the parameters as keys
# 			x: Floating point tensor with shape (N,2)
# 		"""
# 		infectious_pressure = self.infectious_pressure(parameters, x)

# 		K_t_n_SI = 1 - tf.exp(-tf.exp(parameters["log_delta"])*infectious_pressure/tf.cast(self.N, tf.float32) - tf.exp(parameters["log_epsilon"]))
# 		K_t_n_SI = tf.expand_dims(K_t_n_SI, axis = -1)
# 		K_t_n_S  = tf.concat((1-K_t_n_SI, K_t_n_SI, tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
# 		K_t_n_S  = tf.expand_dims(K_t_n_S, axis = -2)

# 		quarantine_rate = tf.ones(tf.shape(x[...,1]))*tf.exp(parameters["log_gamma"])

# 		K_t_n_IN = 1 - tf.exp(-quarantine_rate)
# 		K_t_n_IN = tf.expand_dims(K_t_n_IN, axis = -1)  
# 		K_t_n_I  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), K_t_n_IN, 1-K_t_n_IN, tf.zeros(tf.shape(K_t_n_SI))), axis = -1)
# 		K_t_n_I  = tf.expand_dims(K_t_n_I, axis = -2)

# 		removal_prob = tf.ones(tf.shape(x[...,1]))

# 		K_t_n_NR = removal_prob
# 		K_t_n_NR = tf.expand_dims(K_t_n_NR, axis = -1)  
# 		K_t_n_N  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), 1 - K_t_n_NR, K_t_n_NR), axis = -1)
# 		K_t_n_N  = tf.expand_dims(K_t_n_N, axis = -2)

# 		K_t_n_R  = tf.concat((tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.zeros(tf.shape(K_t_n_SI)), tf.ones(tf.shape(K_t_n_SI))), axis = -1)
# 		K_t_n_R  = tf.expand_dims(K_t_n_R, axis = -2)

# 		return tf.concat((K_t_n_S, K_t_n_I, K_t_n_N, K_t_n_R), axis  = -2)
    
# 	def G_t(self, parameters):
# 		"""
# 		The stochastic matrix computing the reporting probabilities.
# 		Each row refer to a different state, while the columns refer
# 		to the reporting states, with the first one being unreported

# 		Args:
# 			parameters: a dictionary with the parameters as keys
# 		"""

# 		G_t_reported = tf.expand_dims(tf.linalg.diag(tf.math.sigmoid(parameters["logit_prob_testing"])), axis =0)*tf.ones((self.N, self.M, self.M))

# 		return tf.concat((1 - tf.reduce_sum(G_t_reported, axis = -1, keepdims=True), G_t_reported), axis = -1)