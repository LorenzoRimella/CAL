import os
import argparse

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import time

import sys
sys.path.append('CAL/Scripts/')
from model import *
from CAL import *

name_simulation = "FM_HMC_inference_informative_mixed_"

input_path  = "CAL/Data/FM/"

output_path = "CAL/Data/FM/HMC_prior/"

if not os.path.exists(output_path):
	os.makedirs(output_path)
	os.makedirs(output_path+"Check/")

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

########################################
# Load parameters, covariates and locations

local_autorities_covariates  = tf.convert_to_tensor(np.load(input_path+"local_autorities_covariates.npy"), dtype = tf.float32)
farms_covariates = tf.convert_to_tensor(np.load(input_path+"farms_covariates.npy"), dtype = tf.float32)
Y = tf.convert_to_tensor(np.load(input_path+"cut_Y_FM.npy"), dtype = tf.float32)

communities = farms_covariates[:,-1:]
farms_covariates = farms_covariates[:,:2]

SIR = FM_SIR(local_autorities_covariates, communities, farms_covariates)

learning_parameters = {
			"log_tau":1,
			"log_beta":1, "b_S":2, "b_I":2, "log_phi":1,	
			"log_gamma":1,
			"log_rho":1, "log_psi":1,
			"log_epsilon":1,
			"logit_prob_I_testing":1}

list_best_param = []
list_best_loss = []

best_initial = []

input_path_2  = "rerun/"

name_simulation_load = "FM_inference_next2"

dict_best_param = {}

for key in learning_parameters.keys():

	optim_parameters_file_name = input_path+input_path_2+name_simulation_load+"_parameters_"+key+".npy"
	optim_parameters = np.load(optim_parameters_file_name)

	dict_best_param[key] = optim_parameters

loss_file_name       = "FM_inference_next2_loss.npy"
loss       = np.load(input_path+input_path_2+loss_file_name, allow_pickle=False)

best_parameters = {}
for key in learning_parameters.keys():

	best_parameters[key] = np.concatenate(dict_best_param[key][np.where(np.all(loss!=0, axis = 1))[0],...], axis =0)[-1]

tf.random.set_seed((123))
np.random.seed((123))

seed_setting, seed_carry = tfp.random.split_seed( (123), n = 2, salt = name_simulation)

parameters_tensor = best_parameters
parameters_vector_HMC = tf.convert_to_tensor([parameters_tensor["log_tau"][0],
					      parameters_tensor["log_beta"][0],
					      parameters_tensor["b_S"][0], parameters_tensor["b_S"][1],
					      parameters_tensor["b_I"][0], parameters_tensor["b_I"][1],
                                              parameters_tensor["log_phi"][0],
                               		      parameters_tensor["log_gamma"][0],
                               		      parameters_tensor["log_rho"][0],
                               		      parameters_tensor["log_psi"][0],
                               		      parameters_tensor["log_epsilon"][0],
					      parameters_tensor["logit_prob_I_testing"][0]], 
					      dtype = tf.float32)


string_par = ["Parameter "+str(parameters_vector_HMC.numpy()), "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

# https://www.gov.uk/government/publications/foot-and-mouth-disease-control-strategy/foot-and-mouth-disease-control-strategy-for-great-britain
parameter_dim = tf.shape(parameters_vector_HMC)[0]

def prior_fn():
    
	prior_mean = tf.convert_to_tensor([	12.5,
						0.0,
						0.0, 0.0,
						0.0, 0.0,
						0.0,
						3,
						12.5,
						1.2,
						0.0,
						0.0], 
						dtype = tf.float32)
    
	prior_var = tf.convert_to_tensor([	2,      # 0.25,  # 
						100.0,
						100.0, 100.0,
						100.0, 100.0,
						100.0,
						0.5,    # 0.25,  # 
						2,      # 0.25,  # 
						0.25,   # 0.25,  # 
						100.0,
						100.0], 
						dtype = tf.float32)

    
	return tfd.MultivariateNormalDiag(loc = prior_mean, scale_diag = prior_var)

def log_likelihood_fn(parameters_vector_HMC):

	parameters_HMC = {
			"log_tau": parameters_vector_HMC[0:1],
			"log_beta": parameters_vector_HMC[1:2],
			"b_S": parameters_vector_HMC[2:4], 
			"b_I": parameters_vector_HMC[4:6], 
                        "log_phi": parameters_vector_HMC[6:7],
                        "log_gamma": parameters_vector_HMC[7:8],
                        "log_rho": parameters_vector_HMC[8:9],
                        "log_psi": parameters_vector_HMC[9:10],
                        "log_epsilon": parameters_vector_HMC[10:11],
			"logit_prob_I_testing": parameters_vector_HMC[11:], 	
	}

	_, _, log_likelihood = CAL(SIR, parameters_HMC, Y)

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

adaptive_step_size = 0.00274625

seed_mcmc, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_mcmc")

samples, kernel_results = run_hmc(parameters_vector_HMC, adaptive_step_size, seed_mcmc)

posterior_samples = samples.numpy()

for reps in range(200):
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