# Constraint-Guided Causal Analysis Using Bayesian Networks for Medical Diagnosis

This project implements a complete constraint-guided causal analysis pipeline using Bayesian networks on the ASIA medical diagnosis benchmark. The goal is to study how data-driven structure learning and expert knowledge can be combined to recover medically meaningful graph structures and support downstream probabilistic and causal reasoning.

The pipeline compares PC, Hill Climb, and GES structure learning algorithms against the ground-truth ASIA DAG. After evaluating graph recovery using skeleton and orientation-based F1 scores, the best baseline model is refined using expert knowledge. The project then estimates conditional probability distributions using Maximum Likelihood Estimation, saves fitted Bayesian network models, performs probabilistic inference using Variable Elimination, checks causal identification through adjustment-set matching, and evaluates interventional causal inference using do-operator queries.


## Main Features

- Ground-truth ASIA Bayesian network visualization
- Discrete Bayesian network modeling for categorical medical variables
- Structure learning using PC, Hill Climb, and GES
- Two-level graph recovery evaluation using Skeleton F1 and Orientation F1
- Expert-knowledge-guided Hill Climb using required, forbidden, temporal, and basic/custom constraints
- Parameter estimation using Maximum Likelihood Estimation
- Saving and loading fitted Bayesian network models
- Probabilistic inference using Variable Elimination
- Marginal, joint, and MAP diagnostic queries
- Causal identification using adjustment-set matching
- Interventional causal inference using do-operator queries
- Result comparison across PC, Hill Climb, GES, and Hill Climb + Basic Expert models


## Project Pipeline

1. Load and visualize the ground-truth ASIA Bayesian network.
2. Generate simulated samples from the ASIA benchmark.
3. Learn DAG structures using PC, Hill Climb, and GES.
4. Estimate CPDs for each learned DAG using Maximum Likelihood Estimation.
5. Save the fitted Bayesian network models.
6. Evaluate learned graphs using Skeleton F1 and Orientation F1.
7. Select the best baseline model for expert knowledge integration.
8. Apply expert knowledge to Hill Climb structure learning.
9. Perform probabilistic inference using Variable Elimination.
10. Perform causal identification using adjustment-set matching.
11. Perform interventional causal inference using do-operator queries.

## Main Contribution

This project does not propose a new causal discovery algorithm. Instead, it builds and evaluates a complete Bayesian-network-based causal analysis workflow for medical diagnosis. The main contribution is the integration of structure learning, expert-knowledge-guided refinement, parameter estimation, probabilistic inference, causal identification, and interventional causal inference in one reproducible pipeline.
