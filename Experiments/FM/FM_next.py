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

name_simulation = "FM_inference_next"

input_path  = "CAL/Data/FM/"

output_path = "CAL/Data/FM/rerun/"

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

seed_setting, seed_carry = tfp.random.split_seed( (123), n = 2, salt = name_simulation)

string_par = ["Start", "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

learning_parameters = {
			"log_tau":1,
			"log_beta":1, "b_S":2, "b_I":2, "log_phi":1,	
			"log_gamma":1,
			"log_rho":1, "log_psi":1,
			"log_epsilon":1,
			"logit_prob_I_testing":1}

tf.random.set_seed((123))
np.random.seed((123))

par_to_upd = best_parameters

string_par = ["At the beginning we had a loglikelihood of "+str(CAL_compiled(SIR, par_to_upd, Y)[2].numpy()), "\n"]
f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.close()
	
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

n_gradient_steps = 100
n_rerun = 3


loss_numpy = np.zeros((n_rerun, n_gradient_steps+1))
parameters_numpy = {}
for key in learning_parameters.keys():

	parameters_numpy[key] = np.zeros((n_rerun, n_gradient_steps+1, learning_parameters[key]))


for r in range(n_rerun):

	loss_tensor, parameters_tensor = CAL_inference(SIR, par_to_upd, Y, learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

	loss_numpy[r,:] = loss_tensor

	for key in learning_parameters.keys():
		
		parameters_numpy[key][r,...] = parameters_tensor[key]
		par_to_upd[key] = parameters_tensor[key][-1,...]


	string_par = ["At the end of the "+str(r)+" rerun we had a loglikelihood of "+str(CAL_compiled(SIR, par_to_upd, Y)[2].numpy()), "\n"]
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(optimizer.get_state())
	f.writelines(["####################################", "\n"])
	f.close()

	np.save(output_path+name_simulation+"_loss.npy",       loss_numpy)

	for key in learning_parameters.keys():
		np.save(output_path+name_simulation+"_parameters_"+key+".npy", parameters_numpy[key])
