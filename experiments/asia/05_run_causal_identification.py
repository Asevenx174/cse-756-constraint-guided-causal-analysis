from src.simulation.asia import load_asia_model, simulate_asia_samples
from src.structure_learning.constraint_based import learn_pc_discrete
from src.identification.causal_identification import compare_adjustment_sets


def print_comparison_result(title, comparison):
    print(f"\n{title}")
    print("-" * len(title))

    print("Exposure:", comparison["exposure"])
    print("Outcome:", comparison["outcome"])

    print("\nTrue DAG")
    print("Success:", comparison["true_success"])
    print("Adjustment set:", comparison["true_adjustment_set"])

    print("\nLearned DAG")
    print("Success:", comparison["learned_success"])
    print("Adjustment set:", comparison["learned_adjustment_set"])

    print("\nComparison")
    print("Same adjustment set:", comparison["same_adjustment_set"])
    print("Missing from learned:", comparison["missing_from_learned"])
    print("Extra in learned:", comparison["extra_in_learned"])


def main():
    true_model = load_asia_model()
    asia_samples = simulate_asia_samples(n_samples=10000)

    exposure = "smoke"
    outcome = "lung"

    learned_dag = learn_pc_discrete(
        samples=asia_samples,
        return_type="dag",
    )

    comparison = compare_adjustment_sets(
        true_model=true_model,
        learned_dag=learned_dag,
        exposure=exposure,
        outcome=outcome,
    )

    print_comparison_result(
        title="Causal Identification Comparison: true DAG vs learned DAG",
        comparison=comparison,
    )


if __name__ == "__main__":
    main()