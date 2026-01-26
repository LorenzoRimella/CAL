import os
import argparse

import numpy as np

import pickle

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import time

import sys
sys.path.append('CAL/Scripts/')
from model import *
from model_overdispersed import *
from CAL import *
from CAL_overdispersed import *

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

name_simulation = "FM_SMC_ovd"+str(task_id)

input_path  = "CAL/Data/FM/"

output_path = "CAL/Data/FM/SMC_ovd/"

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

	best_parameters[key] = tf.convert_to_tensor(np.concatenate(dict_best_param[key][np.where(np.all(loss!=0, axis = 1))[0],...], axis =0)[-1], dtype = tf.float32)

tf.random.set_seed((123))
np.random.seed((123))

ovd_prior = {}
ovd_prior["log_Xi"] = tfp.distributions.Normal(loc = 0.0, scale = 0.25)

seed_smc = tfp.random.split_seed( 123, n = 10, salt='smc')

simulation_result = {}
simulation_result["shared"] = {}
simulation_result["local"] = {}

simulation_result["local"]["params"]  = [] 
simulation_result["local"]["ESS"]     = []
simulation_result["local"]["loglike"] = []
simulation_result["local"]["time"] = []

simulation_result["shared"]["params"]  = [] 
simulation_result["shared"]["ESS"]     = []
simulation_result["shared"]["loglike"] = []
simulation_result["shared"]["time"] = []


for exp in range(10):

	seed_smc_1, seed_smc_2 = tfp.random.split_seed( seed_smc[exp], n = 2, salt='smc_split')

	ovd_SIR = ovd_FM_SIR(local_autorities_covariates, communities, farms_covariates, ovd_prior)

	string_par = ["Start Alg 1", "\n"]

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.close()

	def blocking_function_1(ovd_ibm, log_likelihood_increment):

		return tf.reduce_sum(log_likelihood_increment, axis = -1, keepdims = True)

	def unblocking_function_1(ovd_ibm, indeces):

		return tf.einsum("...c,cn->...n", indeces, tf.ones((1, ovd_ibm.N), tf.int32))

	start =  time.time()
	parameters_SMC_1, log_likelihood_test_1, ESS_1 = ovd_CALSMC_likelihood(ovd_SIR, best_parameters, Y, blocking_function_1, unblocking_function_1, n_particles = 1024, seed_smc = seed_smc_1)
	simulation_result["shared"]["time"].append((time.time() - start))

	simulation_result["shared"]["params"].append(parameters_SMC_1) 
	simulation_result["shared"]["ESS"].append(log_likelihood_test_1)
	simulation_result["shared"]["loglike"].append(ESS_1)

	np.save(output_path+name_simulation+"_output_SMC_ovd.npy", simulation_result)

	string_par = ["Start Alg 2", "\n"]

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.close()


	ovd_SIR = ovd_FM_SIR_communities(local_autorities_covariates, communities, farms_covariates, ovd_prior)

	def blocking_function_2(ovd_ibm, log_likelihood_increment):

		return tf.einsum("...n,nc->...c", log_likelihood_increment, ovd_ibm.communities)

	def unblocking_function_2(ovd_ibm, indeces):

		return tf.einsum("...c,nc->...n", indeces, tf.cast(ovd_ibm.communities, tf.int32))

	start =  time.time()
	parameters_SMC_2, log_likelihood_test_2, ESS_2 = ovd_CALSMC_likelihood(ovd_SIR, best_parameters, Y, blocking_function_2, unblocking_function_2, n_particles = 512, seed_smc = seed_smc_2)
	simulation_result["local"]["time"].append((time.time() - start))
	
	simulation_result["local"]["params"].append(parameters_SMC_2) 
	simulation_result["local"]["ESS"].append(log_likelihood_test_2)
	simulation_result["local"]["loglike"].append(ESS_2)

	with open(output_path+name_simulation+"_output_SMC_ovd.pkl", "wb") as f:
		pickle.dump(simulation_result, f)

	# np.save(output_path+name_simulation+"_output_SMC_ovd.npy", simulation_result)
