import os
import argparse

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import time

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

import sys
sys.path.append('CAL/Scripts/')
from FM_model import *
from CAL import *

name_simulation = "FM"

input_path  = "CAL/Data/FM/"

output_path = "CAL/Data/FM/Output_"+str(task_id)+"/"

if not os.path.exists(output_path):
	os.makedirs(output_path)
	os.makedirs(output_path+"Check/")

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

########################################
# Load parameters, covariates and locations

indexes = tf.convert_to_tensor(np.load(input_path+"indexes_FM_cumbria.npy"), dtype = tf.int64)
values  = tf.convert_to_tensor(np.load(input_path+"values_FM_cumbria.npy"), dtype = tf.float32)
covariates = tf.convert_to_tensor(np.load(input_path+"covariates_FM_cumbria.npy"), dtype = tf.float32)

Y = tf.convert_to_tensor(np.load(input_path+"Y_FM_cumbria.npy"), dtype = tf.float32)
time_before_infection = tf.cast(tf.shape(Y)[0], tf.float32) - tf.reduce_sum(tf.math.cumsum( Y[...,3], axis = 0), axis = 0)

string_par = ["Start", "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

FM_ibm = sparse_FM_SINR(values, indexes, covariates)

learning_parameters = {#"logit_prior_infection":tf.shape(logit(tf.math.exp(-time_before_infection/5))).numpy()[0], 
		       "log_tau":1,
		       "log_delta":1, "log_zeta":1, "log_xi":1, "log_chi":1, "log_psi":1, "log_gamma":1,"log_epsilon":1}

n_gradient_steps = 10000
n_initial_conditions = 10

loss_numpy = np.zeros((n_initial_conditions, n_gradient_steps+1))
parameters_numpy = {}
for key in learning_parameters.keys():

	if key == "logit_prior_infection":
		parameters_numpy[key] = np.zeros((n_initial_conditions, learning_parameters[key]))

	else:
		parameters_numpy[key] = np.zeros((n_initial_conditions, n_gradient_steps+1, learning_parameters[key]))

for i in range(n_initial_conditions):
	par_to_upd = {#"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5 + np.random.normal(loc = 0.0, scale = np.std(time_before_infection/5)))),
		"log_tau":tf.convert_to_tensor([np.log(40) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_delta":tf.convert_to_tensor([np.log(0.001) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_zeta":tf.convert_to_tensor([np.log(100) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_xi":tf.convert_to_tensor([np.log(10.0) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_chi":tf.convert_to_tensor([np.log(0.6) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_psi":tf.convert_to_tensor([np.log(2.0) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_gamma":tf.convert_to_tensor([np.log(0.25) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"log_epsilon":tf.convert_to_tensor([np.log(0.0001) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
		"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}
	
	while tf.math.is_nan(CAL_compiled(FM_ibm, par_to_upd, Y)[2]):
		par_to_upd = {#"logit_prior_infection":logit(tf.math.exp(-time_before_infection/5 + np.random.normal(loc = 0.0, scale = np.std(time_before_infection/5)))),
		        "log_tau":tf.convert_to_tensor([np.log(40) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_delta":tf.convert_to_tensor([np.log(0.001) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_zeta":tf.convert_to_tensor([np.log(100) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_xi":tf.convert_to_tensor([np.log(10.0) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_chi":tf.convert_to_tensor([np.log(0.6) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_psi":tf.convert_to_tensor([np.log(2.0) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_gamma":tf.convert_to_tensor([np.log(0.25) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"log_epsilon":tf.convert_to_tensor([np.log(0.0001) + np.random.normal(loc = 0.0, scale = 1)], dtype = tf.float32),
			"logit_prob_testing":logit(tf.convert_to_tensor([0.0, 0.0, 1.0, 0.0], dtype = tf.float32)),}

	string_par = ["At the beginning of iteration "+str(i)+" we had a loglikelihood of "+str(CAL_compiled(FM_ibm, par_to_upd, Y)[2].numpy()), "\n"]
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.close()
	
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

	loss_tensor, parameters_tensor = CAL_inference(FM_ibm, par_to_upd, Y[:100,...], learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

	loss_numpy[i,:] = loss_tensor

	for key in learning_parameters.keys():
		
		if key == "logit_prior_infection":
			parameters_numpy[key][i,...] = parameters_tensor[key][-1,...]

		else:
			parameters_numpy[key][i,...] = parameters_tensor[key]

	string_par = ["At the end of "+str(i)+" we had a loglikelihood of "+str(CAL_compiled(FM_ibm, par_to_upd, Y)[2].numpy()), "\n"]
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.close()

	np.save(output_path+name_simulation+"_loss.npy",       loss_numpy)

	for key in learning_parameters.keys():
		np.save(output_path+name_simulation+"_parameters_"+key+".npy", parameters_numpy[key])