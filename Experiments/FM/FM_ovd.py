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
from model_overdispersed import *
from CAL import *
from CAL_overdispersed import *

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

name_simulation = "FM_SMC_ovd_"+str(task_id)

input_path  = "CAL/Data/FM/"

output_path = "CAL/Data/FM/SMC/"

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

for test in range(10):
	temp_output_path = input_path+"Output_"+str(test)+"/"
	loss_file_name       = "FM_loss.npy"

	loss       = np.load(temp_output_path+loss_file_name,       allow_pickle=False)
	best_index = np.argmin( np.where(np.isnan(loss[:,-1]), 100, loss[:,-1]))

	list_best_loss.append(loss[best_index,-1])

	dict_best_param = {}

	for i in range(int(len(learning_parameters.keys())/2)):
		for j in range(2):

			key = list(learning_parameters.keys())[2*i+j]

			optim_parameters_file_name = "FM_parameters_"+key+".npy"
			optim_parameters = np.load(temp_output_path+optim_parameters_file_name)

			dict_best_param[key] = tf.convert_to_tensor(optim_parameters[best_index,-1,:], dtype = tf.float32)

	list_best_param.append(dict_best_param)

best_optim = np.argmin(np.stack(list_best_loss))

best_parameters = list_best_param[best_optim]

tf.random.set_seed((123))
np.random.seed((123))

grid_mean = [-2, -1, -0.5, -0.25, 0.0, 0.25, 0.5, 1, 2]
grid_std  = [0, 0.125, 0.25, 0.5, 1, 2, 5]

current_mean = grid_mean[task_id]

parameters_SMC_list = []
ESS_list = []
log_likelihood_test_list = []

seed_smc = tfp.random.split_seed( 123, n = len(grid_std), salt='smc')

counter = 0
for std_ovd in grid_std:

	ovd_prior = {"log_Xi": tfp.distributions.Normal(loc = current_mean, scale = std_ovd)}
	ovd_SIR = ovd_FM_SIR(local_autorities_covariates, communities, farms_covariates, ovd_prior)

	string_par = ["Start", "\n"]

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.writelines(["####################################", "\n"])
	f.close()

	def blocking_function(ovd_ibm, log_likelihood_increment):

		return tf.reduce_sum(log_likelihood_increment, axis = -1, keepdims = True)

	def unblocking_function(ovd_ibm, indeces):

		return tf.einsum("...c,cn->...n", indeces, tf.ones((1, ovd_ibm.N), tf.int32))


	parameters_SMC, log_likelihood_test, ESS = ovd_CALSMC_likelihood(ovd_SIR, best_parameters, Y, blocking_function, unblocking_function, n_particles = 1024, seed_smc = seed_smc[counter])

	parameters_SMC_list.append(parameters_SMC)
	ESS_list.append(ESS)
	log_likelihood_test_list.append(log_likelihood_test)
	
	counter = counter+1

import pickle

with open(output_path+name_simulation+"_paramSMC.pkl", "wb") as f:
    pickle.dump(parameters_SMC, f)

with open(output_path+name_simulation+"_paramESS.pkl", "wb") as f:
    pickle.dump(ESS, f)

with open(output_path+name_simulation+"_loglike.pkl", "wb") as f:
    pickle.dump(log_likelihood_test, f)

# np.save(output_path+name_simulation+"_paramSMC.npy", parameters_SMC)
# np.save(output_path+name_simulation+"_paramESS.npy", ESS)
# np.save(output_path+name_simulation+"_loglike.npy",  log_likelihood_test)
