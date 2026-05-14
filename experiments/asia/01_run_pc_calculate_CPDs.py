from src.simulation.asia import load_asia_model, simulate_asia_samples
from src.structure_learning.constraint_based import learn_pc_discrete
from src.evaluation.graph_metrics import evaluate_graph_recovery
from src.visualization.graphviz_plot import save_graphviz_dag
from src.estimation.parameter_estimation import estimate_parameters_mle
from src.expert_knowledge.asia import get_asia_expert_knowledge

def main():
    true_model = load_asia_model()
    asia_samples = simulate_asia_samples(n_samples=10000)
    
      # Default: no expert knowledge.
    # Caution !If you make it false then you will override the non-expert-guided dag
    use_expert_knowledge = True
    knowledge_type = "basic"
    
    expert_knowledge = get_asia_expert_knowledge(
        use_expert_knowledge=use_expert_knowledge,
        knowledge_type=knowledge_type,
    )

    # To enable expert knowledge, change:
    # use_expert_knowledge = True
    #
    # Available knowledge_type values:
    # "basic"
    # "forbidden"
    # "required"

    learned_dag = learn_pc_discrete(
        samples=asia_samples,
        expert_knowledge=expert_knowledge
    )
    
    scores = evaluate_graph_recovery(
        estimated_model=learned_dag,
        true_model=true_model,
        algorithm_name="PC stable chi_square",
        print_scores=True
    )
    
    if use_expert_knowledge:
        graph_output_path = f"results/graphs/asia_pc_learned_expert_{knowledge_type}.png"
    else:
        graph_output_path = "results/graphs/asia_pc_learned.png"
        
    save_graphviz_dag(
        model_or_graph=learned_dag,
        output_path=graph_output_path,
        layout="dot",
    )
    
    print("\n All scores:")
    print(scores)
    
    print("\nParameter Estimation: MLE")
    print("-------------------------")

    estimate_parameters_mle(
        learned_dag=learned_dag,
        samples=asia_samples,
        print_cpds=True,
    )
    
if __name__ == "__main__":
    main()
    
    
