from pathlib import Path

from pgmpy.inference import CausalInference

from src.persistence.model_io import load_model_pickle


def run_causal_inference_for_model(model_path, model_name):
    fitted_model = load_model_pickle(model_path)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("\nModel valid:")
    print(fitted_model.check_model())

    print("\nModel edges:")
    print(list(fitted_model.edges()))

    causal_inference = CausalInference(fitted_model)

    print("\nCausal / Interventional Inference")
    print("---------------------------------")

    result = causal_inference.query(
        variables=["dysp"],
        do={"smoke": "yes"},
        show_progress=False,
    )

    print("\nP(dysp | do(smoke=yes))")
    print(result)

    result = causal_inference.query(
        variables=["xray"],
        do={"either": "yes"},
        show_progress=False,
    )

    print("\nP(xray | do(either=yes))")
    print(result)

    result = causal_inference.query(
        variables=["dysp"],
        do={"smoke": "yes"},
        evidence={"xray": "yes"},
        show_progress=False,
    )

    print("\nP(dysp | do(smoke=yes), xray=yes)")
    print(result)

    result = causal_inference.query(
    variables=["bronc"],
    do={"smoke": "yes"},
    show_progress=False,
)

    print("\nP(bronc | do(smoke=yes))")
    print(result)

    result = causal_inference.query(
    variables=["dysp"],
    do={"smoke": "yes"},
    show_progress=False,
)

    print("\nP(dysp | do(smoke=yes))")
    print(result)

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

        run_causal_inference_for_model(
            model_path=model_path,
            model_name=item["model_name"],
        )


if __name__ == "__main__":
    main()