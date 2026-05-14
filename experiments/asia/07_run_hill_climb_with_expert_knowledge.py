from src.simulation.asia import load_asia_model, simulate_asia_samples
from src.structure_learning.score_based import learn_hill_climb_discrete
from src.evaluation.graph_metrics import (
    evaluate_graph_recovery
)
from src.visualization.graphviz_plot import save_graphviz_dag
from src.estimation.parameter_estimation import estimate_parameters_mle
from src.expert_knowledge.asia import get_asia_expert_knowledge
from src.persistence.model_io import save_model_pickle


def main():
    true_model = load_asia_model()

    n_samples = 10000
    seed = 42

    asia_samples = simulate_asia_samples(
        n_samples=n_samples,
        seed=seed,
    )
    # keep it always true
    use_expert_knowledge = True
    knowledge_type = "basic"

    expert_knowledge = get_asia_expert_knowledge(
        use_expert_knowledge=use_expert_knowledge,
        knowledge_type=knowledge_type,
    )

    learned_dag = learn_hill_climb_discrete(
        samples=asia_samples,
        expert_knowledge=expert_knowledge,
    )

    algorithm_name = f"Hill Climb with expert knowledge: {knowledge_type}"

    scores = evaluate_graph_recovery(
        estimated_model=learned_dag,
        true_model=true_model,
        algorithm_name=algorithm_name,
        print_scores=True,
    )

    run_name = f"asia_hill_climb_expert_{knowledge_type}"

    graph_output_path = f"results/graphs/{run_name}.png"
    model_output_path = f"results/models/{run_name}_fitted.pkl"

    save_graphviz_dag(
        model_or_graph=learned_dag,
        output_path=graph_output_path,
        layout="dot",
    )

    print("\nGraph saved to:")
    print(graph_output_path)

    print("\nAll scores:")
    print(scores)

    print("\nParameter Estimation: MLE")
    print("-------------------------")

    fitted_model = estimate_parameters_mle(
        learned_dag=learned_dag,
        samples=asia_samples,
        print_cpds=True,
    )

    save_model_pickle(
        model=fitted_model,
        output_path=model_output_path,
    )

    print("\nFitted expert-guided model saved to:")
    print(model_output_path)


if __name__ == "__main__":
    main()