import os
import argparse

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import time

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

import sys
sys.path.append('CAL/Scripts/')
from model import *
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

local_autorities_covariates  = tf.convert_to_tensor(np.load(input_path+"local_autorities_covariates.npy"), dtype = tf.float32)
farms_covariates = tf.convert_to_tensor(np.load(input_path+"farms_covariates.npy"), dtype = tf.float32)
Y = tf.convert_to_tensor(np.load(input_path+"cut_Y_FM.npy"), dtype = tf.float32)

communities = farms_covariates[:,-1:]
farms_covariates = farms_covariates[:,:2]

SIR = FM_SIR(local_autorities_covariates, communities, farms_covariates)

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

n_gradient_steps = 10000
n_initial_conditions = 10

loss_numpy = np.zeros((n_initial_conditions, n_gradient_steps+1))
parameters_numpy = {}
for key in learning_parameters.keys():

	parameters_numpy[key] = np.zeros((n_initial_conditions, n_gradient_steps+1, learning_parameters[key]))


tf.random.set_seed((123+task_id))
np.random.seed((123+task_id))

seed_setting = tfp.random.split_seed( (123+task_id), n = n_initial_conditions, salt = name_simulation+str(task_id))

for i in range(n_initial_conditions):

	seed_current = tfp.random.split_seed( seed_setting[i], n = 11, salt = name_simulation+str(task_id))
	seed_carry   = seed_current[-1]
	
	par_to_upd = {
			"log_tau":       tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[0], sample_shape= (1)),
			"log_beta":      tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[1], sample_shape= (1)),
			"b_S":                       tf.convert_to_tensor([+0.35, 0.25], dtype = tf.float32)  + tfp.distributions.Normal(loc = 0.0, scale = 0.1).sample(seed = seed_current[2], sample_shape= (2)),
			"b_I":                       tf.convert_to_tensor([+0.35, 0.25], dtype = tf.float32)  + tfp.distributions.Normal(loc = 0.0, scale = 0.1).sample(seed = seed_current[3], sample_shape= (2)),
			"log_phi":       tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[4], sample_shape= (1)),
			"log_gamma":     tf.math.log(tf.convert_to_tensor([0.5],         dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[5], sample_shape= (1)),
			"log_rho":       tf.math.log(tf.convert_to_tensor([5],           dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[6], sample_shape= (1)),
			"log_psi":       tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[7], sample_shape= (1)),
			"log_epsilon":   tf.math.log(tf.convert_to_tensor([0.00005],     dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 2.0).sample(seed = seed_current[8], sample_shape= (1)),
			"logit_prob_I_testing":logit(tf.convert_to_tensor([0.3],         dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[9], sample_shape= (1))}
	
	while tf.math.is_nan(CAL_compiled(SIR, par_to_upd, Y)[2]):

		seed_current = tfp.random.split_seed( seed_carry, n = 11, salt = name_simulation+str(task_id))
		seed_carry   = seed_current[-1]

		par_to_upd = {
				"log_tau":       tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[0], sample_shape= (1)),
				"log_beta":      tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[1], sample_shape= (1)),
				"b_S":                       tf.convert_to_tensor([+0.35, 0.25], dtype = tf.float32)  + tfp.distributions.Normal(loc = 0.0, scale = 0.1).sample(seed = seed_current[2], sample_shape= (2)),
				"b_I":                       tf.convert_to_tensor([+0.35, 0.25], dtype = tf.float32)  + tfp.distributions.Normal(loc = 0.0, scale = 0.1).sample(seed = seed_current[3], sample_shape= (2)),
				"log_phi":       tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[4], sample_shape= (1)),
				"log_gamma":     tf.math.log(tf.convert_to_tensor([0.5],         dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[5], sample_shape= (1)),
				"log_rho":       tf.math.log(tf.convert_to_tensor([5],           dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[6], sample_shape= (1)),
				"log_psi":       tf.math.log(tf.convert_to_tensor([10],          dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[7], sample_shape= (1)),
				"log_epsilon":   tf.math.log(tf.convert_to_tensor([0.00005],     dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 2.0).sample(seed = seed_current[8], sample_shape= (1)),
				"logit_prob_I_testing":logit(tf.convert_to_tensor([0.3],         dtype = tf.float32)) + tfp.distributions.Normal(loc = 0.0, scale = 1.0).sample(seed = seed_current[9], sample_shape= (1))}

	string_par = ["At the beginning of iteration "+str(i)+" we had a loglikelihood of "+str(CAL_compiled(SIR, par_to_upd, Y)[2].numpy()), "\n"]
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.close()
	
	optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

	loss_tensor, parameters_tensor = CAL_inference(SIR, par_to_upd, Y, learning_parameters, optimizer, n_gradient_steps, initialization = "parameters")

	loss_numpy[i,:] = loss_tensor

	for key in learning_parameters.keys():
		
		parameters_numpy[key][i,...] = parameters_tensor[key]

	string_par = ["At the end of "+str(i)+" we had a loglikelihood of "+str(CAL_compiled(SIR, par_to_upd, Y)[2].numpy()), "\n"]
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.close()

	np.save(output_path+name_simulation+"_loss.npy",       loss_numpy)

	for key in learning_parameters.keys():
		np.save(output_path+name_simulation+"_parameters_"+key+".npy", parameters_numpy[key])