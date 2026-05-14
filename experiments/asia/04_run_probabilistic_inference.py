from pathlib import Path

from src.persistence.model_io import load_model_pickle
from src.inference.probabilistic_inference import (
    query_probability,
    query_map_assignment,
)


def run_probabilistic_inference_for_model(model_path, model_name):
    fitted_model = load_model_pickle(model_path)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("\nModel valid:")
    print(fitted_model.check_model())

    print("\nModel edges:")
    print(list(fitted_model.edges()))

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


def main():
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

        run_probabilistic_inference_for_model(
            model_path=model_path,
            model_name=item["model_name"],
        )


if __name__ == "__main__":
    main()