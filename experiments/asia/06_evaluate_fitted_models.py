from pathlib import Path

from src.simulation.asia import load_asia_model
from src.persistence.model_io import load_model_pickle
from src.evaluation.graph_metrics import (
    evaluate_graph_recovery,
    evaluate_edge_orientation,
)


def evaluate_fitted_model(model_path, true_model, algorithm_name):
    fitted_model = load_model_pickle(model_path)

    skeleton_scores = evaluate_graph_recovery(
        estimated_model=fitted_model,
        true_model=true_model,
        algorithm_name=algorithm_name,
        print_scores=True,
    )

    orientation_scores = evaluate_edge_orientation(
        estimated_model=fitted_model,
        true_model=true_model,
        algorithm_name=algorithm_name,
        print_scores=True,
    )

    return {
        "algorithm": algorithm_name,
        "skeleton_f1": skeleton_scores["skeleton_f1"],
        "orientation_precision": orientation_scores["orientation_precision"],
        "orientation_recall": orientation_scores["orientation_recall"],
        "orientation_f1": orientation_scores["orientation_f1"],
    }


def main():
    true_model = load_asia_model()

    fitted_models = [
        {
            "path": "results/models/asia_pc_fitted.pkl",
            "algorithm_name": "PC fitted model",
        },
        {
            "path": "results/models/asia_hill_climb_fitted.pkl",
            "algorithm_name": "Hill Climb fitted model",
        },
        {
            "path": "results/models/asia_ges_fitted.pkl",
            "algorithm_name": "GES fitted model",
        },
        {
            "path": "results/models/asia_hill_climb_expert_basic_fitted.pkl",
            "algorithm_name": "Hill Climb expert fitted model",
        },
    ]

    all_scores = []

    for item in fitted_models:
        model_path = Path(item["path"])

        if not model_path.exists():
            print("\nSkipping missing model:")
            print(model_path)
            continue

        print("\n" + "=" * 70)

        scores = evaluate_fitted_model(
            model_path=model_path,
            true_model=true_model,
            algorithm_name=item["algorithm_name"],
        )

        all_scores.append(scores)

    print("\n" + "=" * 70)
    print("Fitted Model Score Summary")
    print("--------------------------")

    for scores in all_scores:
        print(
        scores["algorithm"],
        "| Skeleton F1:",
        scores["skeleton_f1"],
        "| Orientation F1:",
        scores["orientation_f1"],
    )


if __name__ == "__main__":
    main()