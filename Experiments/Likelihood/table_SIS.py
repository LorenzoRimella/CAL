import os
import argparse

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import pickle as pkl

import time

import sys
sys.path.append('CAL/Scripts/')
from model import *
from CAL import *
from smc import *
from block_smc import *

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

N_list = [10, 100, 1000, 10000]
N      = N_list[task_id]

batch_size_list = [1, 10, 100, 100]
b = batch_size_list[task_id]

name_simulation = "likelihood_SIS_"+str(N)+"_V2"

input_path  = "CAL/Data/Likelihood/Input/SIS/"
output_path = "CAL/Data/Likelihood/Output/SIS_V2/"+str(N)+"/"

replicates = 100
if not os.path.exists(output_path):
	os.makedirs(output_path)
	os.makedirs(output_path+"Check/")

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(["Start Experiment", "\n"])
f.writelines(["####################################", "\n"])
f.writelines(["####################################", "\n"])
f.close()

M = 2
T = 100

parameters = {"prior_infection":tf.convert_to_tensor([1-0.01, 0.01], dtype = tf.float32),
	      "log_beta":tf.math.log(tf.convert_to_tensor([0.2], dtype = tf.float32)),
              "b_I":tf.convert_to_tensor([+0.3], dtype = tf.float32),
              "b_S":tf.convert_to_tensor([-0.3], dtype = tf.float32),
	      "log_gamma":tf.math.log(tf.convert_to_tensor([0.1], dtype = tf.float32)),
              "b_R":tf.convert_to_tensor([+0.2], dtype = tf.float32),
              "logit_sensitivity":logit(
                tf.convert_to_tensor([0.9], dtype = tf.float32)),
              "logit_specificity":logit(
                tf.convert_to_tensor([0.95], dtype = tf.float32)),
              "logit_prob_testing":logit(
                tf.convert_to_tensor([0.2, 0.5], dtype = tf.float32)),}

covariates = tf.convert_to_tensor(np.load(input_path+"covariates.npy"), dtype=tf.float32)[:N,:]

Y_SIS_N = tf.convert_to_tensor(np.load(input_path+"Y_"+str(N)+".npy"), dtype=tf.float32)[:N,:]

SIS_N    = logistic_SIS(covariates)

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["Data Loaded", "\n"])
f.writelines(["####################################", "\n"])
f.close()

table_N = {}

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["CAL", "\n"])
f.close()

table_N["CAL"] = {}

start =  time.time()
table_N["CAL"]["log_like"] = CAL(SIS_N, parameters, Y_SIS_N)[-1]
CAL_time = time.time() - start
table_N["CAL"]["comp_time"] = CAL_time

loglikelihood_test_run_SIS = CAL_compiled(SIS_N, parameters, Y_SIS_N)[-1]

table_N["CAL_compiled"] = {}
start =  time.time()
table_N["CAL_compiled"]["log_like"]  = CAL_compiled(SIS_N, parameters, Y_SIS_N)[-1]
CAL_time = time.time() - start
table_N["CAL_compiled"]["comp_time"] = CAL_time

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["SMC", "\n"])
f.close()

P_list = [256, 512, 1024, 2048, 4096, 8192]

tf.random.set_seed((123+N))
np.random.seed((123+N))

seed_carry, _ = tfp.random.split_seed( 123+N, n = 2, salt = "smc_initial_split")

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["APF", "\n"])
f.close()

table_N["APF"] = {}

for P in P_list:
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["P="+str(P), "\n"])
	f.close()

	table_N["APF"][P] = {}

	table_N["APF"][P]["log_like"] = []
	table_N["APF"][P]["comp_time"] = []
	for i in range(replicates):
		seed_P_next, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "smc_P_initial_split")

		start =  time.time()
		table_N["APF"][P]["log_like"].append(apf(SIS_N, parameters, Y_SIS_N, P=P, seed_smc = seed_P_next))
		bapf_time = time.time() - start
		table_N["APF"][P]["comp_time"].append(bapf_time)

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["Save data", "\n"])
	f.close()

	with open(output_path+"/"+name_simulation+".pkl", "wb") as f:
		pkl.dump(table_N, f)

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["block APF", "\n"])
f.close()

table_N["block_APF"] = {}

for P in P_list:
	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["P="+str(P), "\n"])
	f.close()

	table_N["block_APF"][P] = {}

	if np.exp(np.log(N) + 2*np.log(P) + np.log(M*4) - np.log(1e9))<18: # avoid OOM error

		table_N["block_APF"][P]["log_like"] = []
		table_N["block_APF"][P]["comp_time"] = []
		for i in range(replicates):
			seed_P_next, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "smc_P_initial_split")

			start =  time.time()
			table_N["block_APF"][P]["log_like"].append(bapf(SIS_N, parameters, Y_SIS_N, P=P, seed_smc = seed_P_next))
			bapf_time = time.time() - start
			table_N["block_APF"][P]["comp_time"].append(bapf_time)

	else:
		table_N["block_APF"][P]["log_like"] = "Out of memory"
		table_N["block_APF"][P]["comp_time"] = np.nan


	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["Save data", "\n"])
	f.close()

	with open(output_path+"/"+name_simulation+".pkl", "wb") as f:
		pkl.dump(table_N, f)

f= open(output_path+"Check/"+name_simulation+".txt", "a")
f.writelines(["####################################", "\n"])
f.writelines(["block APF batched", "\n"])
f.close()

table_N["block_APF_batched"] = {}

for P in P_list:

	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["P="+str(P), "\n"])
	f.close()

	table_N["block_APF_batched"][P] = {}

	if np.exp(np.log(b) + 2*np.log(P) + np.log(M) - np.log(1e9))<18:

		table_N["block_APF_batched"][P]["log_like"] = []
		table_N["block_APF_batched"][P]["comp_time"] = []

		for i in range(replicates):
			seed_P_next, seed_carry = tfp.random.split_seed( seed_carry, n = 2, salt = "smc_P_initial_split")

			start =  time.time()
			table_N["block_APF_batched"][P]["log_like"].append(batch_bapf(SIS_N, parameters, Y_SIS_N, P=P, batch_size = b, seed_smc = seed_P_next))
			bapf_time = time.time() - start
			table_N["block_APF_batched"][P]["comp_time"].append(bapf_time)

			with open(output_path+"/"+name_simulation+".pkl", "wb") as f:
				pkl.dump(table_N, f)

			f= open(output_path+"Check/"+name_simulation+".txt", "a")
			f.writelines(["replicate="+str(i), "\n"])
			f.close()

	else:
		table_N["block_APF_batched"][P]["log_like"] = "Out of memory"
		table_N["block_APF_batched"][P]["comp_time"] = np.nan


	f= open(output_path+"Check/"+name_simulation+".txt", "a")
	f.writelines(["Save data", "\n"])
	f.close()