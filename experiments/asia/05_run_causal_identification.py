from pathlib import Path

from src.simulation.asia import load_asia_model
from src.persistence.model_io import load_model_pickle
from src.identification.causal_identification import compare_adjustment_sets


def print_comparison_result(title, comparison):
    print(f"\n{title}")
    print("-" * len(title))

    print("Exposure:", comparison["exposure"])
    print("Outcome:", comparison["outcome"])

    print("\nTrue DAG")
    print("Success:", comparison["true_success"])

    if comparison["true_success"]:
        print("Adjustment set:", comparison["true_adjustment_set"])
    else:
        print("Adjustment set: Not identifiable")

    print("\nLearned / Fitted Model DAG")
    print("Success:", comparison["learned_success"])

    if comparison["learned_success"]:
        print("Adjustment set:", comparison["learned_adjustment_set"])
    else:
        print("Adjustment set: Not identifiable")
        print("Message: This learned graph could not identify a valid adjustment set.")

    print("\nComparison")

    if comparison["true_success"] and comparison["learned_success"]:
        print("Same adjustment set:", comparison["same_adjustment_set"])
        print("Missing from learned:", comparison["missing_from_learned"])
        print("Extra in learned:", comparison["extra_in_learned"])
    else:
        print("Same adjustment set: Not comparable")
        print("Reason: Identification failed in true DAG or learned DAG.")


def main():
    true_model = load_asia_model()

    causal_questions = [
        {
            "exposure": "smoke",
            "outcome": "lung",
        },
        {
            "exposure": "smoke",
            "outcome": "bronc",
        },
        {
            "exposure": "either",
            "outcome": "xray",
        },
        {
            "exposure": "either",
            "outcome": "dysp",
        },
    ]

    fitted_models = [
        {
            "path": "results/models/asia_pc_fitted.pkl",
            "model_name": "PC fitted model",
        },
        {
            "path": "results/models/asia_hill_climb_fitted.pkl",
            "model_name": "Hill Climb fitted model",
        },
        {
            "path": "results/models/asia_ges_fitted.pkl",
            "model_name": "GES fitted model",
        },
        {
            "path": "results/models/asia_hill_climb_expert_basic_fitted.pkl",
            "model_name": "Hill Climb expert basic fitted model",
        },
    ]

    for item in fitted_models:
        model_path = Path(item["path"])

        if not model_path.exists():
            print("\nSkipping missing model:")
            print(model_path)
            continue

        fitted_model = load_model_pickle(model_path)

        print("\n" + "=" * 80)
        print(item["model_name"])
        print("=" * 80)

        for question in causal_questions:
            exposure = question["exposure"]
            outcome = question["outcome"]

            try:
                comparison = compare_adjustment_sets(
                    true_model=true_model,
                    learned_dag=fitted_model,
                    exposure=exposure,
                    outcome=outcome,
                )

                print_comparison_result(
                    title=f"Causal Identification: {exposure} -> {outcome}",
                    comparison=comparison,
                )

            except Exception as error:
                print(f"\nCausal Identification: {exposure} -> {outcome}")
                print("-" * len(f"Causal Identification: {exposure} -> {outcome}"))
                print("Status: Failed")
                print("Message: Identification could not be completed for this graph.")
                print("Reason:", error)


if __name__ == "__main__":
    main()