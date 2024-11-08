import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd

def sample_cities_on_unit(n_cities, seed_cities, centroids = None):
    
    if centroids == None:
        cities = tfp.distributions.Uniform(low = [0.0, 0.0], high = [1.0, 1.0]).sample(n_cities, seed = seed_cities)

    else:
        return centroids

    return cities

def sample_individuals_from_cities(cities_coords, cities_std, nr_individuals_per_city, seed_cities_indiv):
    
    individuals_rv = tfp.distributions.MultivariateNormalDiag(loc = cities_coords, scale_diag = cities_std)

    return individuals_rv.sample(nr_individuals_per_city, seed = seed_cities_indiv)

def sample_covariates(covariates_rv_list, nr_individuals, seed_covariates):
    
    seed_split = tfp.random.split_seed( seed_covariates, n=len(covariates_rv_list), salt='sim_covariates')

    covariates_list = [covariates_rv_list[i].sample((nr_individuals), seed = seed_split[i]) for i in range(len(covariates_rv_list))]
    covariates_list = [tf.cast(covariates_list[i], tf.float32) for i in range(len(covariates_list))]

    return tf.stack(covariates_list, axis = 1)

def make_covariates(n_cities, nr_individuals_per_city, cities_std, centroids, covariates_rv_list, seed_to_set):

    seed_split = tfp.random.split_seed( seed_to_set, n=3, salt='sim_covariates')
    
    cities_coords = sample_cities_on_unit(n_cities, seed_split[0], centroids)

    coords_individuals_batched = sample_individuals_from_cities(cities_coords, cities_std, nr_individuals_per_city, seed_split[1])
    coords_individuals = tf.reshape(coords_individuals_batched, (tf.reduce_prod(tf.shape(coords_individuals_batched)[:-1]), tf.shape(coords_individuals_batched)[2]))

    nr_individuals = tf.shape(coords_individuals)[0]
    nr_covariates  = len(covariates_rv_list)
    covariates_individuals = sample_covariates(covariates_rv_list, nr_individuals, seed_split[2] )

    data_frame = pd.DataFrame({"x":coords_individuals[:,0], "y":coords_individuals[:,1]})
                              
    for i in range(nr_covariates):
        data_frame["covariate"+str(i)] = covariates_individuals[:,i]

    return data_frame

class stochasticBlockModel:
    
    def __init__(self, B):

        self.B = B

    def pop_sample(self, initial_distribution, N):

        Erv        = tfp.distributions.OneHotCategorical(probs = initial_distribution)

        return tf.cast(Erv.sample(N), dtype = tf.float32)

    def sample(self, E): 

        M = tf.einsum("nj,kj->nk", tf.einsum("nk,kj->nj", E, self.B), E)

        A = tfp.distributions.Binomial(1, probs = tf.experimental.numpy.triu(M, k = 1)).sample()

        return A + tf.transpose(tf.experimental.numpy.triu(A, k = 1))

# if __name__ == "__main__":
#     print("Debugging")
#     seed_to_use = 42

#     tf.random.set_seed((seed_to_use))
#     np.random.seed((seed_to_use))

#     seed_individuals = tfp.random.split_seed( seed_to_use, n=5, salt='seed_synthetic_pop')
#     seed_carry       = seed_individuals[4]

#     n_cities = 10

#     cities_std = tfp.distributions.Gamma(concentration = 10, rate = 120).sample((n_cities, 2), seed = seed_individuals[0])

#     centroids = tfp.distributions.Uniform(low = [0.0, 0.0], high = [1.0, 1.0]).sample(n_cities, seed = seed_individuals[1])

#     nr_individuals_per_city = 100

#     categorical_prob = tfp.distributions.Dirichlet(concentration=tf.ones(10)).sample(seed = seed_individuals[3])
#     covariates_rv_list = [tfp.distributions.Normal(loc = 0.0, scale = 1.0), 
#                         tfp.distributions.Gamma(concentration= 5, rate = 1),
#                         tfp.distributions.Categorical(probs = categorical_prob)]

#     covariates_df = make_covariates(n_cities, nr_individuals_per_city, cities_std, centroids, covariates_rv_list, seed_individuals[2])
