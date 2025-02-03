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

N_list = [500, 5000, 50000]
N      = N_list[task_id]

name_simulation = "graph_inference_"+str(N)

input_path  = "CAL/Data/GraphInference/Input/"

output_path = "CAL/Data/GraphInference/Output/"+str(N)+"/"


if not os.path.exists(output_path):
	os.makedirs(output_path)
	os.makedirs(output_path+"Check/")

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

########################################
# Load parameters, covariates and locations

parameters = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
	      "log_beta":tf.convert_to_tensor([2.5], dtype = tf.float32),
	      "b_S":tf.convert_to_tensor([+0.5], dtype = tf.float32),
              "b_I":tf.convert_to_tensor([+1.0], dtype = tf.float32),
              "log_gamma":tf.convert_to_tensor([0.1], dtype = tf.float32),
              "b_R":tf.convert_to_tensor([-0.5], dtype = tf.float32),
              "log_phi":tf.math.log(
                tf.convert_to_tensor([1], dtype = tf.float32)),
              "log_epsilon":tf.math.log(tf.convert_to_tensor([0.0001], dtype = tf.float32)),
              "logit_sensitivity":logit(
                tf.convert_to_tensor([0.9], dtype = tf.float32)),
              "logit_specificity":logit(
                tf.convert_to_tensor([0.95], dtype = tf.float32)),
              "logit_prob_testing":logit(
                tf.convert_to_tensor([0.2, 0.5], dtype = tf.float32))}

centroids  = np.load(input_path+"Input/centroids.npy")
city_index = np.load(input_path+"Input/city_index.npy")

index             = tf.convert_to_tensor(np.load(input_path+"Input/reshuffle.npy")[:N], dtype = tf.int32)
covariates_sample = tf.convert_to_tensor(np.load(input_path+"Input/covariates.npy"), dtype = tf.float32)

covariates  = tf.gather(covariates_sample, index, axis = 0)
communities = tf.gather(city_index, index, axis = 0)

T = 200
n_covergage = 100
n_gradient_steps = 1000
n_trial = 10

tf.random.set_seed((123+N+T+task_id))
np.random.seed((123+N+T+task_id))

seed_setting = tfp.random.split_seed( (123+N+T+task_id), n = n_covergage, salt = name_simulation+str(task_id))

string_par = ["Population "+str(N), "\n"]

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(string_par)
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

learning_parameters = {"log_beta":1, "b_S":1, "b_I":1, "log_gamma":1, "b_R":1, "log_phi":1, "logit_prob_testing":2} 

true_loss_numpy  = np.zeros(n_covergage)
optim_loss_numpy = np.zeros((n_covergage, n_trial, n_gradient_steps+1))
optim_parameters_numpy = {}
for key in learning_parameters.keys():
	optim_parameters_numpy[key] = np.zeros((n_covergage, n_trial, n_gradient_steps+1, learning_parameters[key]))

for i in range(n_covergage):

	string_par = ["Simulation "+str(i), "\n"]

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["####################################", "\n"])
	f.writelines(string_par)
	f.writelines(["####################################", "\n"])
	f.close()

	########################################
	# Generate data

	SIS = sbm_SIS(communities, centroids, covariates)

	seed_sim, seed_carry = tfp.random.split_seed( seed_setting[i], n = 2, salt = "seed_for_simulation")
	X, Y = simulator(SIS, parameters, T, seed_sim = seed_sim)

	while tf.reduce_sum(X[...,1])<2*T:
		seed_sim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_simulation")
		X, Y = simulator(SIS, parameters, T, seed_sim = seed_sim)

	X, Y = X[:,0,...], Y[:,0,...]

	true_loss = CAL_loss(SIS, parameters, Y)
	true_loss_numpy[i] = true_loss.numpy()

	########################################
	# Optimization
	for j in range(n_trial):
		optimizer = tf.keras.optimizers.Adam(learning_rate = 0.2)


		parameters_to_learn = { "prior_infection":parameters["prior_infection"],
					"log_epsilon":parameters["log_epsilon"],
					"logit_sensitivity":parameters["logit_sensitivity"],
					"logit_specificity":parameters["logit_specificity"]}

		seed_optim, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "seed_for_optimization")
		loss_tensor, parameters_tensor = CAL_inference(SIS, parameters_to_learn, Y, learning_parameters, optimizer, n_gradient_steps, seed_optim, "random")

		optim_loss_numpy[i,j,...] = loss_tensor
		for key in learning_parameters.keys():
			optim_parameters_numpy[key][i,j,...] = parameters_tensor[key]

	np.save(output_path+name_simulation+"_true_loss.npy",        true_loss_numpy)
	np.save(output_path+name_simulation+"_optim_loss.npy",       optim_loss_numpy)
	np.save(output_path+name_simulation+"_optim_parameters.npy", optim_parameters_numpy)