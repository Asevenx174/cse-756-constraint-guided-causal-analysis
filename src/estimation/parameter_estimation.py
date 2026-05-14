from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork


def estimate_parameters_mle(
    learned_dag,
    samples,
    print_cpds=False,
):
    """
    Estimate CPDs for a learned discrete Bayesian network using MLE.
    """
    fitted_model = DiscreteBayesianNetwork()
    fitted_model.add_nodes_from(learned_dag.nodes())
    fitted_model.add_edges_from(learned_dag.edges())

    estimator = MaximumLikelihoodEstimator(
        model=fitted_model,
        data=samples,
    )

    cpds = estimator.get_parameters()
    fitted_model.add_cpds(*cpds)

    print("MLE fitted model valid:", fitted_model.check_model())

    if print_cpds:
        print("\nLearned CPDs:")
        print("-------------")

        for cpd in fitted_model.get_cpds():
            print(cpd)
            print()

    return fitted_model