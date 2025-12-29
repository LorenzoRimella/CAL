import tensorflow as tf

class IBM_AR_common():

	def __init__(self, covariates):

		self.covariates = covariates

	def IBM_log_likelihood(self, parameters, Y):

		p_logistic_1t = tf.math.sigmoid(parameters["logit_lambda"])/(1+tf.exp(-parameters["gamma_Y"]*Y[:-1,...] - tf.einsum("c,nc->n", parameters["gamma_W"], self.covariates)))

		p_logistic_0  = tf.expand_dims( tf.math.sigmoid(parameters["logit_lambda"])/(1+tf.exp(-tf.einsum("c,nc->n", parameters["gamma_W"], self.covariates))), axis = 0)
		p_logistic    = tf.concat((p_logistic_0, p_logistic_1t), axis = 0)

		AR_likelihood_nt = Y_logistic*p_logistic + (1-Y)*(1-p_logistic)

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

		AR_likelihood_nt = Y_logistic*p_logistic + (1-Y)*(1-p_logistic)

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

if __name__ == "__main__":

	import time

	from model_overdispersed import *
	from model import *
	from CAL import *

	from scipy.spatial.distance import pdist, squareform

	import matplotlib.pyplot as plt

	local_autorities_covariates  = tf.convert_to_tensor(np.load("CAL/Data/FM/local_autorities_covariates.npy"), dtype = tf.float32)
	farms_covariates = tf.convert_to_tensor(np.load("CAL/Data/FM/farms_covariates.npy"), dtype = tf.float32)
	Y = tf.convert_to_tensor(np.load("CAL/Data/FM/cut_Y_FM.npy"), dtype = tf.float32)

	communities = farms_covariates[:,-1:]
	farms_covariates = farms_covariates[:,:2]

	SIR = FM_SIR(local_autorities_covariates, communities, farms_covariates)
	local_autorities_covariates_to_farms = tf.einsum("cw,nc->nw", local_autorities_covariates[:,-1:], SIR.communities )

	logist_regression_covariates = tf.concat((farms_covariates, local_autorities_covariates_to_farms), axis =  -1)
	logist_regression_covariates = tf.concat((tf.expand_dims(tf.ones(tf.shape(logist_regression_covariates)[0]), axis = -1), logist_regression_covariates), axis =-1)
	
	Y_logistic = tf.reduce_sum(Y[...,1:], axis = -1)

	IBM_AR = IBM_AR_common(logist_regression_covariates)

	n_gradient_steps = 10000
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
	parameters_to_learn = {"logit_lambda": logit(tf.convert_to_tensor([0.5], dtype = tf.float32)), "gamma_Y":tf.convert_to_tensor([0.1], dtype = tf.float32), "gamma_W":tf.convert_to_tensor([0.5, 0.5, 0.5, 0.5], dtype = tf.float32)}
	print(logistic_log_likelihood(IBM_AR, parameters_to_learn, Y_logistic))
	loss_history_community, parameters_history_community, parameters_trained_community = IBM_AR_inference(IBM_AR, parameters_to_learn, Y_logistic, n_gradient_steps, optimizer)
	print(logistic_log_likelihood(IBM_AR, parameters_trained_community, Y_logistic))

	IBM_AR_community = IBM_AR_local(logist_regression_covariates, SIR.communities)
	
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
	parameters_to_learn = { "logit_lambda": logit(tf.convert_to_tensor([0.5], dtype = tf.float32))*tf.ones(tf.shape(IBM_AR_community.communities)[1]), 
				"gamma_Y":tf.convert_to_tensor([0.1], dtype = tf.float32)*tf.ones(tf.shape(IBM_AR_community.communities)[1]), 
				"gamma_W":tf.convert_to_tensor([0.5, 0.5, 0.5, 0.5], dtype = tf.float32)*tf.ones((tf.shape(IBM_AR_community.communities)[1], 1))}
	print(logistic_log_likelihood(IBM_AR_community, parameters_to_learn, Y_logistic))
	loss_history_community, parameters_history_community, parameters_trained_community = IBM_AR_inference(IBM_AR_community, parameters_to_learn, Y_logistic, n_gradient_steps, optimizer)
	print(logistic_log_likelihood(IBM_AR_community, parameters_trained_community, Y_logistic))

	print("hello")

# def logistic_AR_smc(covariates, parameters, y, P=1024, seed_smc = None):
    
# 	T = tf.shape(y)[0]
# 	C = tf.shape(logist_regression_covariates)[1]
	
# 	seed_smc_run = tfp.random.split_seed( seed_smc, n = T+1, salt = "seed_smc")

# 	beta_0 = tfp.distributions.Normal(loc = tf.zeros(1, tf.float32), scale = tf.math.exp(parameters["log_sigma"])).sample((P, C), seed = seed_smc_run[0])[...,0]

# 	log_likelihood_increment_0 = tf.zeros(1)
# 	ESS_0 = tf.zeros(1)

# 	def body(input, t):

# 		beta_ptm1, _, _ = input

# 		seed_smc_pred, seed_smc_res = tfp.random.split_seed( seed_smc_run[t], n = 2, salt = "seed_smc_body")

# 		beta_pt = tfp.distributions.Normal(loc = parameters["gamma"]*beta_ptm1, scale = tf.math.exp(parameters["log_sigma"])).sample(seed = seed_smc_pred)

# 		beta_cov_pt = tf.einsum("pc,nc->pn", beta_pt, logist_regression_covariates)

# 		p_logistic_pt = 1/(1+tf.exp(-beta_cov_pt))

# 		w_npt = tf.einsum("n,pn->pn", y[t,...], p_logistic_pt) + tf.einsum("n,pn->pn", (1-y[t,...]), (1-p_logistic_pt))

# 		log_weights_pt = tf.reduce_sum(tf.math.log(w_npt), axis = 1)
# 		log_weights_pt = tf.where( log_weights_pt<-400, -400, log_weights_pt)
		
# 		shifted_weights_p = tf.exp(log_weights_pt-tf.reduce_max(log_weights_pt, axis = 0, keepdims = True))
# 		norm_weights_p = shifted_weights_p/tf.reduce_sum(shifted_weights_p, axis = 0, keepdims=True)

# 		ESS_t = 1/tf.reduce_sum(norm_weights_p*norm_weights_p)

# 		log_likelihood_increment = tf.math.log(tf.reduce_mean(shifted_weights_p, axis =0)) + tf.reduce_max(log_weights_pt, axis = 0)

# 		indeces = tfp.distributions.Categorical(probs = norm_weights_p).sample(P, seed = seed_smc_res)
# 		res_X_pt = tf.gather(beta_pt, indeces, axis = 0)

# 		return res_X_pt, tf.expand_dims(ESS_t, axis = 0), tf.expand_dims(log_likelihood_increment, axis = 0)

# 	Beta, ESS, log_likelihood = tf.scan(body, tf.range(0, T), initializer = (beta_0, ESS_0, log_likelihood_increment_0))

# 	return tf.concat((tf.expand_dims(beta_0, axis = 0), Beta), axis = 0), ESS, log_likelihood


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

# 	plt.plot(tf.reduce_sum(Y, axis = 1)[:100,2])
# 	plt.show()

# 	communities = farms_covariates[:,-1:]
# 	farms_covariates = farms_covariates[:,:2]

# 	current_communities = np.unique(communities)
# 	counter = 0
# 	for i in current_communities:

# 		communities = tf.where(communities==i, counter, communities)

# 		counter = counter + 1

# 	SIR = FM_SIR(local_autorities_covariates, communities, farms_covariates)
# 	local_autorities_covariates_to_farms = tf.einsum("cw,nc->nw", local_autorities_covariates[:,-1:], SIR.communities )

# 	logist_regression_covariates = tf.concat((farms_covariates, local_autorities_covariates_to_farms), axis =  -1)
# 	logist_regression_covariates = tf.concat((tf.expand_dims(tf.ones(tf.shape(logist_regression_covariates)[0]), axis = -1), logist_regression_covariates), axis =-1)

# 	Y_logistic = tf.reduce_sum(Y[...,1:], axis = -1)

# 	T = tf.shape(Y)[0]
# 	C = tf.shape(logist_regression_covariates)[1]

# 	parameters = {"logit_lambda": logit(tf.convert_to_tensor([0.5], dtype = tf.float32)), "gamma":tf.convert_to_tensor([0.1], dtype = tf.float32), "log_sigma":tf.convert_to_tensor(np.log([0.5]), dtype = tf.float32)}
# 	Beta, ESS, log_like = logistic_AR_smc(logist_regression_covariates, parameters, Y_logistic, 8192)

	

# 	print("hello")

	



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

# 	plt.plot(tf.reduce_sum(Y, axis = 1)[:100,2])
# 	plt.show()

# 	communities = farms_covariates[:,-1:]
# 	farms_covariates = farms_covariates[:,:2]

# 	current_communities = np.unique(communities)
# 	counter = 0
# 	for i in current_communities:

# 		communities = tf.where(communities==i, counter, communities)

# 		counter = counter + 1

# 	SIR = FM_SIR(local_autorities_covariates, communities, farms_covariates)
# 	local_autorities_covariates_to_farms = tf.einsum("cw,nc->nw", local_autorities_covariates[:,-1:], SIR.communities )

# 	logist_regression_covariates = tf.concat((farms_covariates, local_autorities_covariates_to_farms), axis =  -1)
# 	logist_regression_covariates = tf.concat((tf.expand_dims(tf.ones(tf.shape(logist_regression_covariates)[0]), axis = -1), logist_regression_covariates), axis =-1)

# 	T = tf.shape(Y)[0]
# 	C = tf.shape(logist_regression_covariates)[1]

# 	parameters = {"logit_lambda": logit(tf.convert_to_tensor([0.5], dtype = tf.float32)), "gamma":tf.convert_to_tensor([0.1], dtype = tf.float32), "log_sigma":tf.convert_to_tensor(np.log([0.5]), dtype = tf.float32)}

# 	Beta = tfp.distributions.Normal(loc = 0, scale = 1).sample((T, C))

# 	Beta_W = tf.einsum("tc,nc->tn", Beta, logist_regression_covariates)

# 	Y_logistic = tf.reduce_sum(Y[...,1:], axis = -1)

# 	def logistic_loss(parameters, Beta, Y_logistic):

# 		N = tf.cast(tf.shape(Y_logistic)[1], dtype = tf.float32)

# 		Beta_W = tf.einsum("tc,nc->tn", Beta, logist_regression_covariates)

# 		p_logistic = tf.math.sigmoid(parameters["logit_lambda"])/(1+tf.exp(-Beta_W))

# 		log_emission = tf.reduce_sum(tfp.distributions.Bernoulli(probs = p_logistic).log_prob(Y_logistic))
		
# 		log_dynamic = tf.reduce_sum(tfp.distributions.Normal(loc = parameters["gamma"]*Beta[:-1,:], scale = tf.math.exp(parameters["log_sigma"])).log_prob(Beta[1:,:]))
# 		log_initial = tf.reduce_sum(tfp.distributions.Normal(loc = tf.zeros(tf.shape(Beta[0:])), scale = tf.math.exp(parameters["log_sigma"])).log_prob(Beta[0,:]))
		
# 		return -(log_initial + log_dynamic + log_emission)/N
	
# 	@tf.function(jit_compile=True)
# 	def grad_Beta_logistic_loss(parameters, Beta, Y_logistic):

# 		with tf.GradientTape() as g:

# 			loss = logistic_loss(parameters, Beta, Y_logistic)

# 		return loss, g.gradient(loss, Beta)
	
# 	@tf.function(jit_compile=True)
# 	def loss_grad_hess_Beta_logistic(parameters, Beta, Y_logistic):

# 		with tf.GradientTape(persistent=True) as outer_tape:

# 			with tf.GradientTape() as inner_tape:

# 				loss = logistic_loss(parameters, Beta, Y_logistic)

# 			grad = inner_tape.gradient(loss, Beta)

# 		hess = outer_tape.jacobian(grad, Beta)

# 		del outer_tape

# 		return loss, grad, hess
	

# 	Beta_to_optim = tf.Variable(Beta)

# 	plt.plot(Beta_to_optim)
# 	plt.show()
# 	print(logistic_loss(parameters, Beta_to_optim, Y_logistic))

# 	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# 	n_gradient_steps = 1000
# 	for i in range(n_gradient_steps):
# 		loss, grad = grad_Beta_logistic_loss(parameters, Beta_to_optim, Y_logistic)
# 		optimizer.apply_gradients(zip(grad, [Beta_to_optim]))

# 	plt.plot(Beta_to_optim)
# 	plt.show()

# 	hessian = tf.cast(tf.shape(Y_logistic)[1], dtype = tf.float32)*loss_grad_hess_Beta_logistic(parameters, Beta_to_optim, Y_logistic)[2]

# 	print("hello")

