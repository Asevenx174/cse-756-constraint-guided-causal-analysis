from pathlib import Path

from src.simulation.asia import load_asia_model
from src.persistence.model_io import load_model_pickle
from src.evaluation.graph_metrics import print_missing_and_extra_edges


def compare_saved_model_edges(model_path, true_model, model_name):
    fitted_model = load_model_pickle(model_path)

    print_missing_and_extra_edges(
        estimated_model=fitted_model,
        true_model=true_model,
        model_name=model_name,
    )


def main():
    true_model = load_asia_model()

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

        compare_saved_model_edges(
            model_path=model_path,
            true_model=true_model,
            model_name=item["model_name"],
        )


if __name__ == "__main__":
    main()