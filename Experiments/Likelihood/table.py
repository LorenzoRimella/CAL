import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import os
import sys
sys.path.append('Scripts/')
from model import *
from CAL import *

replicates = 100

input_path  = "Data/Likelihood/Input/"
output_path = "Data/Likelihood/SIS/"

import time

########################################
# SIS
M = 2
N = 1000
covmean = 0
initial_infection_rate = 0.01
T = 100

covariates = tf.convert_to_tensor(np.load(input_path+"W_1000_numpy.npy"), dtype=tf.float32)
y = tf.convert_to_tensor(np.einsum("ijk->kij", np.load(input_path+"Y_1000_numpy.npy")), dtype=tf.float32)
Y_SIS = tf.concat((tf.expand_dims(tf.where(tf.reduce_sum(y[1:,...], axis = -1)==0, tf.ones(1, dtype = tf.float32), tf.zeros(1, dtype = tf.float32)), axis = -1), y[1:,...]), axis = -1)

N = tf.shape(covariates)[0]

parameters = {"beta_0": tf.convert_to_tensor([-np.log((1/initial_infection_rate)-1), +0], dtype = tf.float32),
              "beta_l": tf.convert_to_tensor([-1.0, +2.0], dtype = tf.float32),
              "beta_g": tf.convert_to_tensor([-1.0, -1.0], dtype = tf.float32),
              "iota": tf.math.log(tf.convert_to_tensor(0.001, dtype = tf.float32 ) ),
               "q":  tf.convert_to_tensor([0.6, 0.4], dtype = tf.float32)
           }


SIS = simba_SIS(covariates)

start =  time.time()
loglikelihood_SIS = CAL_compiled(SIS, parameters, Y_SIS)
print("It took "+str(time.time() - start)+"sec")
print("The loglikelihood is "+str(loglikelihood_SIS[-1].numpy()))

start =  time.time()
loglikelihood_SIS = CAL_compiled(SIS, parameters, Y_SIS)
print("It took "+str(time.time() - start)+"sec")
print("The loglikelihood is "+str(loglikelihood_SIS[-1].numpy()))


########################################
# SEIR
M = 4
initial_infection_rate = 0.01
T = 100

covariates = tf.convert_to_tensor(np.load(input_path+"W_SEIR_numpy.npy"), dtype=tf.float32)
y = tf.transpose(tf.convert_to_tensor(np.load(input_path+"Y_SEIR_numpy.npy"), dtype=tf.float32), (2, 0, 1))
Y_SEIR = tf.concat((tf.expand_dims(tf.where(tf.reduce_sum(y[1:,...], axis = -1)==0, tf.ones(1, dtype = tf.float32), tf.zeros(1, dtype = tf.float32)), axis = -1), y[1:,...]), axis = -1)

N = tf.shape(covariates)[0]

parameters = {"beta_0": tf.convert_to_tensor([-np.log((1/initial_infection_rate)-1), +0], dtype = tf.float32),
              "beta_l": tf.convert_to_tensor([-1.0, +2.0], dtype = tf.float32),
              "beta_g": tf.convert_to_tensor([-1.0, -1.0], dtype = tf.float32),
              "iota": tf.math.log(tf.convert_to_tensor(0.001, dtype = tf.float32 ) ),
	      "rho": tf.math.log(tf.convert_to_tensor(0.2, dtype = tf.float32 ) ),
               "q":  tf.convert_to_tensor([0.0, 0.0, 0.4, 0.6], dtype = tf.float32)
           }


SEIR = simba_SEIR(covariates)

start =  time.time()
loglikelihood_SEIR = CAL_compiled(SEIR, parameters, Y_SEIR)
print("It took "+str(time.time() - start)+"sec")
print("The loglikelihood is "+str(loglikelihood_SEIR[-1].numpy()))

start =  time.time()
loglikelihood_SEIR = CAL_compiled(SEIR, parameters, Y_SEIR)
print("It took "+str(time.time() - start)+"sec")
print("The loglikelihood is "+str(loglikelihood_SEIR[-1].numpy()))

