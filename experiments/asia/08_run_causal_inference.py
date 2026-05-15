from pathlib import Path

from pgmpy.inference import CausalInference

from src.persistence.model_io import load_model_pickle


def print_causal_query(
    causal_inference,
    title,
    query_variables,
    intervention,
    evidence=None,
):
    try:
        result = causal_inference.query(
            variables=query_variables,
            do=intervention,
            evidence=evidence,
            show_progress=False,
        )

        print(f"\n{title}")
        print(result)

    except Exception as error:
        print(f"\n{title}")
        print("Skipped invalid causal query.")
        print("Reason:", error)


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

    causal_questions = [
        {
            "title": "P(dysp | do(smoke=yes))",
            "query_variables": ["dysp"],
            "intervention": {"smoke": "yes"},
            "evidence": None,
        },
        {
            "title": "P(bronc | do(smoke=yes))",
            "query_variables": ["bronc"],
            "intervention": {"smoke": "yes"},
            "evidence": None,
        },
        {
            "title": "P(xray | do(either=yes))",
            "query_variables": ["xray"],
            "intervention": {"either": "yes"},
            "evidence": None,
        },
        {
            "title": "P(dysp | do(smoke=yes), xray=yes)",
            "query_variables": ["dysp"],
            "intervention": {"smoke": "yes"},
            "evidence": {"xray": "yes"},
        },
    ]

    for question in causal_questions:
        print_causal_query(
            causal_inference=causal_inference,
            title=question["title"],
            query_variables=question["query_variables"],
            intervention=question["intervention"],
            evidence=question["evidence"],
        )


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