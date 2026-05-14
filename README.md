# cse-756-constraint-guided-causal-analysis

This project explores constraint-guided causal analysis using Bayesian networks. It applies structure learning algorithms such as PC, Hill Climb, and GES on the ASIA benchmark network, evaluates learned graph structures against the ground-truth DAG, and studies the effect of expert knowledge on causal discovery.

The project also includes parameter estimation using maximum likelihood estimation, saving fitted Bayesian network models, probabilistic inference with Variable Elimination, and interventional causal inference using do-operator queries.

## Main Features

- Ground-truth ASIA DAG visualization
- Structure learning with PC, Hill Climb, and GES
- Expert-knowledge-guided structure learning
- Graph recovery evaluation using skeleton F1-score
- Parameter estimation with learned CPDs
- Saving and loading fitted Bayesian network models
- Probabilistic inference over saved fitted models
- Causal/interventional inference using do-queries