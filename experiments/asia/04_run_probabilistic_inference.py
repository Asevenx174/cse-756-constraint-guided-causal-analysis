from src.simulation.asia import simulate_asia_samples
from src.structure_learning.constraint_based import learn_pc_discrete
from src.estimation.parameter_estimation import estimate_parameters_mle
from src.inference.probabilistic_inference import (
    query_probability,
    query_map_assignment,
)
from src.structure_learning.score_based import (
    learn_hill_climb_discrete,
    learn_ges_discrete,
)


def main():
    asia_samples = simulate_asia_samples(n_samples=10000)

    # Option 1: Constraint-based structure learning using PC.
    learned_dag = learn_pc_discrete(
        samples=asia_samples
    )
    
     # Option 2: Score-based structure learning using Hill Climb.
    # Uncomment this block and comment out the PC block above to use Hill Climb.
    # learned_dag = learn_hill_climb_discrete(
    #     samples=asia_samples
    # )

    # Option 3: Score-based structure learning using GES.
    # Uncomment this block and comment out the PC block above to use GES.
    # learned_dag = learn_ges_discrete(
    #     samples=asia_samples
    # )

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

    joint_result = query_probability(
        fitted_model=fitted_model,
        query_variables=["lung", "bronc"],
        evidence={"dysp": "yes"},
        joint=True,
    )

    print("\nJoint query: P(lung, bronc | dysp=yes)")
    print(joint_result)

    marginal_results = query_probability(
        fitted_model=fitted_model,
        query_variables=["lung", "bronc"],
        evidence={"dysp": "yes"},
        joint=False,
    )

    print("\nSeparate marginal queries with joint=False")
    print("-----------------------------------------")

    for variable, distribution in marginal_results.items():
        print(f"\nP({variable} | dysp=yes)")
        print(distribution)

    map_result = query_map_assignment(
        fitted_model=fitted_model,
        query_variables=["lung", "bronc"],
        evidence={"dysp": "yes"},
    )

    print("\nMAP query: most likely lung and bronc given dysp=yes")
    print(map_result)


if __name__ == "__main__":
    main()