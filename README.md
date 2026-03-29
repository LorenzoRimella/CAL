# Scalable calibration of individual-based epidemic models through categorical approximations

The computational cost of exact likelihood evaluation for partially observed and highly-heterogeneous individual-based models grows exponentially with the population size, therefore inference relies on approximations. Sampling-based approaches to this problem such as Sequential Monte Carlo or Approximate Bayesian Computation usually require simulation of every individual in the population multiple times and are heavily reliant on design of bespoke proposal distributions or summary statistics. We propose a deterministic recursive approach to approximating the likelihood function using categorical distributions. The resulting algorithm has a computational cost as low as linear in the population size and is amenable to automatic differentiation. We prove consistency of the maximum approximate likelihood estimator of model parameters. Despite the simple structure of our approximation, establishing consistency relies on detailed and carefully engineered theoretical characterization of the large-population behavior of the data-generating process and our algorithm. We empirically test our approach on a range of models with various flavors of heterogeneity: different sets of disease states, individual-specific susceptibility and infectivity, spatial interaction mechanisms, under-reporting and mis-reporting. We demonstrate strong calibration performance and computational advantages over competitor methods. We conclude by illustrating the effectiveness of our approach in a real-world large-scale application using Foot-and-Mouth data from the 2001 outbreak in the United Kingdom.

# Repository guidelines

Figures and tutorials:

- figures.ipynb reproduces all the figures in the paper except for those in Sections 5.3 and D.4; see the notebook for further details.
- tutorial_SIR.ipynb explains how to perform a CAL optimization of an SIR agent-based model using our code and how to test it. This also replicates Sections 5.3 and D.4.
- tutorial_HMC.ipynb explains how to run an HMC as in Section 5.1.

Experiments and paper sections:

- Experiments/HMC can be used to reproduce Section 5.1, where Experiments/HMC/HMC_replicates.py is used to check the coverage.
- Experiments/SpatialInference and Experiments/GraphInference can be used to reproduce Sections 5.2 and D.3.
- tutorial_SIR.ipynb can be used to reproduce Sections 5.3 and D.4.
- Experiments/Likelihood/table_SIS_supp.py and Experiments/Likelihood/table_SEIR_supp.py can be used to reproduce Section D.5.
- Experiments/Likelihood/table_SIS.py and Experiments/Likelihood/table_SIS_ju.py can be used to reproduce Sections 5.4 and D.6.
- Experiments/FM is used to reproduce Sections 5.5 and D.7, specifically: FM.py performs the first optimization with 10k Adam steps and different initial conditions, FM_next.py performs the next 400k Adam steps, FM_HMC.py performs the HMC, FM_ovd.py performs the grid search, FM_ovd_rep.py performs the replicates on the shared and local authority overdispersion.
- the end of figures.ipynb reports the benchmarking experiment.