from src.simulation.asia import simulate_asia_samples
from src.structure_learning.constraint_based import learn_pc_discrete
from src.estimation.parameter_estimation import estimate_parameters_mle
from src.inference.probabilistic_inference import query_probability


def main():
    asia_samples = simulate_asia_samples(n_samples=10000)

    learned_dag = learn_pc_discrete(
        samples=asia_samples
    )

    fitted_model = estimate_parameters_mle(
        learned_dag=learned_dag,
        samples=asia_samples,
        print_cpds=False,
    )

    print("\nProbabilistic Inference")
    print("-----------------------")

    result = query_probability(
        fitted_model=fitted_model,
        query_variables=["dysp"],
        evidence={"smoke": "yes"},
    )

    print("\nP(dysp | smoke=yes)")
    print(result)

    result = query_probability(
        fitted_model=fitted_model,
        query_variables=["xray"],
        evidence={"either": "yes"},
    )

    print("\nP(xray | either=yes)")
    print(result)


if __name__ == "__main__":
    main()