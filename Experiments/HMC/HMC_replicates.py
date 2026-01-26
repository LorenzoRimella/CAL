import os
import argparse

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import time

import sys
sys.path.append('CAL/Scripts/')
from model import *
from CAL import *

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

N = 1000 

name_simulation = "HMC_inference_"+str(N)+"_"+str(task_id)

input_path  = "CAL/Data/HMC/Input/"

output_path = "CAL/Data/HMC/Output/"+str(N)+"_"+str(task_id)+"/"

if not os.path.exists(output_path):
	os.makedirs(output_path)
	os.makedirs(output_path+"Check/")

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

########################################
# Load parameters, covariates and locations

tf.random.set_seed((123+N+task_id))
np.random.seed((123+N+task_id))

seed_setting, seed_carry = tfp.random.split_seed( (123+N+task_id), n = 2, salt = name_simulation+str(task_id))

covariates = tf.expand_dims(tfp.distributions.Normal(loc = 0.0, scale = 1).sample(seed = seed_setting, sample_shape= N), axis = -1)

SIS = logistic_SIS(covariates)

parameters = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
	      "log_beta":tf.math.log(
		      tf.convert_to_tensor([0.2], dtype = tf.float32)),
	      "b_S":tf.convert_to_tensor([+0.5], dtype = tf.float32),
              "b_I":tf.convert_to_tensor([+1.0], dtype = tf.float32),
              "log_gamma":tf.math.log(
		      tf.convert_to_tensor([0.1], dtype = tf.float32)),
              "b_R":tf.convert_to_tensor([-0.5], dtype = tf.float32),
              "logit_sensitivity":logit(
                tf.convert_to_tensor([0.9], dtype = tf.float32)),
              "logit_specificity":logit(
                tf.convert_to_tensor([0.95], dtype = tf.float32)),
              "logit_prob_testing":logit(
                tf.convert_to_tensor([0.2, 0.5], dtype = tf.float32))}

T    = 200

seed_sim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_simulation")
X, Y = simulator(SIS, parameters, T, seed_sim = seed_sim)

while tf.reduce_sum(X[...,1])<2*T:
	seed_sim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_simulation")
	X, Y = simulator(SIS, parameters, T, seed_sim = seed_sim)

X, Y = X[:,0,...], Y[:,0,...]

string_par = ["Adam warm-start", "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

tfd = tfp.distributions

loss_list = []
parameters_list = []

for i in range(100):
	seed_initial, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_initialization")

	parameters_vector_HMC = tf.convert_to_tensor([np.log(0.1), 0.0, 0.0, np.log(0.1), 0.0, logit(0.1), logit(0.1)], dtype = tf.float32) # this is not used though (just for the shape)
	parameters_vector_HMC = tfd.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_initial, sample_shape= tf.shape(parameters_vector_HMC))

	parameters_HMC = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
		"log_beta":parameters_vector_HMC[0:1],
		"b_I":parameters_vector_HMC[1:2],
		"b_S":parameters_vector_HMC[2:3],
		"log_gamma":parameters_vector_HMC[3:4],
		"b_R":parameters_vector_HMC[4:5],
		"logit_sensitivity": tf.convert_to_tensor([logit(0.9)], dtype = tf.float32), #parameters_vector_HMC[5:6],
		"logit_specificity": tf.convert_to_tensor([logit(0.95)], dtype = tf.float32), #parameters_vector_HMC[6:7],
		"logit_prob_testing":parameters_vector_HMC[5:7],}
	
	try_index = 0
	while (tf.math.is_nan(CAL_loss(SIS, parameters_HMC, Y)) or CAL_loss(SIS, parameters_HMC, Y)>10*T) and try_index<10:

		seed_initial, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_initialization")
		parameters_vector_HMC = tfd.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_initial, sample_shape= tf.shape(parameters_vector_HMC))

		parameters_HMC = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
			"log_beta":parameters_vector_HMC[0:1],
			"b_I":parameters_vector_HMC[1:2],
			"b_S":parameters_vector_HMC[2:3],
			"log_gamma":parameters_vector_HMC[3:4],
			"b_R":parameters_vector_HMC[4:5],
			"logit_sensitivity": tf.convert_to_tensor([logit(0.9)], dtype = tf.float32), #parameters_vector_HMC[5:6],
			"logit_specificity": tf.convert_to_tensor([logit(0.95)], dtype = tf.float32), #parameters_vector_HMC[6:7],
			"logit_prob_testing":parameters_vector_HMC[5:7],}
		
		try_index = try_index + 1

	n_gradient_steps = 1000
	learning_parameters = {"log_beta":1, "b_I":1, "b_S":1, 
			"log_gamma":1,"b_R":1, 
			"logit_prob_testing":2, }

	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

	loss_tensor, parameters_tensor = CAL_inference(SIS, parameters_HMC, Y, learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

	loss_list.append(loss_tensor)
	parameters_list.append(parameters_tensor)

string_par = ["Adam warm-start finished", "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

loss_stack = tf.stack(loss_list, axis = 0)[:,-1]
best_index = tf.math.argmin(tf.where(tf.math.is_nan(loss_stack), 10000*tf.ones(tf.shape(loss_stack)), loss_stack))

parameters_tensor = parameters_list[best_index]

parameters_vector_HMC = tf.convert_to_tensor([parameters_tensor["log_beta"][-1,0],
                                              parameters_tensor["b_I"][-1,0], parameters_tensor["b_S"][-1,0],
                               		      parameters_tensor["log_gamma"][-1,0], parameters_tensor["b_R"][-1,0],
					      parameters_tensor["logit_prob_testing"][-1,0], parameters_tensor["logit_prob_testing"][-1,1]], dtype = tf.float32)


string_par = ["Parameter "+str(parameters_vector_HMC.numpy()), "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

parameter_dim = tf.shape(parameters_vector_HMC)[0]

def prior_fn():
    
    return tfd.MultivariateNormalDiag(loc=tf.zeros([parameter_dim]), scale_diag=100*tf.ones([parameter_dim]))

def log_likelihood_fn(parameters_vector_HMC):

	parameters_HMC = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
		"log_beta":parameters_vector_HMC[0:1],
		"b_I":parameters_vector_HMC[1:2],
		"b_S":parameters_vector_HMC[2:3],
		"log_gamma":parameters_vector_HMC[3:4],
		"b_R":parameters_vector_HMC[4:5],
		"logit_sensitivity": tf.convert_to_tensor([logit(0.9)], dtype = tf.float32), #parameters_vector_HMC[5:6],
		"logit_specificity": tf.convert_to_tensor([logit(0.95)], dtype = tf.float32), #parameters_vector_HMC[6:7],
		"logit_prob_testing":parameters_vector_HMC[5:7],}

	_, _, log_likelihood = CAL(SIS, parameters_HMC, Y)

	return log_likelihood

def joint_log_prob_fn(parameters_vector_HMC):
    
    prior = prior_fn().log_prob(parameters_vector_HMC)
    likelihood = log_likelihood_fn(parameters_vector_HMC)
    
    return prior + likelihood

@tf.function(jit_compile=True)
def run_hmc(initial_state, adaptive_step_size, seed_to_use = None):
    
    # Create the HMC kernel
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=joint_log_prob_fn,
        step_size=adaptive_step_size,
        num_leapfrog_steps=10)
    
    # Run the MCMC chain
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=1000,  # Number of posterior samples
        num_burnin_steps=0,  # Burn-in steps
        current_state=initial_state,
        kernel=hmc_kernel,
	seed = seed_to_use,
        trace_fn=lambda _, pkr: pkr)  # Trace kernel results for diagnostics
    
    return samples, kernel_results

adaptive_step_size = 0.01

seed_mcmc, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_mcmc")

samples, kernel_results = run_hmc(parameters_vector_HMC, adaptive_step_size, seed_mcmc)

posterior_samples = samples.numpy()

for reps in range(100):
	np.save(output_path+name_simulation+"_posterior.npy", posterior_samples)

	# string_par = ["Acceptance rate: "+str(kernel_results.inner_results.is_accepted.numpy().mean()), "\n"]
	string_par = ["Acceptance rate: "+str(kernel_results.is_accepted.numpy().mean()), "\n"]
	string_par_2 = ["Step size: "+str(adaptive_step_size), "\n"]

	if (kernel_results.is_accepted.numpy().mean()<0.55):
		adaptive_step_size = adaptive_step_size*0.65
		
	if (kernel_results.is_accepted.numpy().mean()>0.75):
		adaptive_step_size = adaptive_step_size*1.35

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(string_par_2)
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.close()

	seed_mcmc, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_mcmc")

	samples, kernel_results = run_hmc(samples[-1,:], adaptive_step_size, seed_mcmc)

	posterior_samples = np.concatenate((posterior_samples, samples.numpy()), axis =0)

string_par = ["Acceptance rate: "+str(kernel_results.is_accepted.numpy().mean()), "\n"]
string_par_2 = ["Step size: "+str(adaptive_step_size), "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(string_par_2)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

np.save(output_path+name_simulation+"_posterior.npy", posterior_samples)