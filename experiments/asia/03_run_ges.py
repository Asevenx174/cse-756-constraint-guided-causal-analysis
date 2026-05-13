from src.simulation.asia import load_asia_model, simulate_asia_samples
from src.structure_learning.score_based import learn_ges_discrete 
from src.evaluation.graph_metrics import evaluate_graph_recovery
from src.visualization.graphviz_plot import save_graphviz_dag

def main():
    true_model = load_asia_model()
    asia_samples = simulate_asia_samples(n_samples=10000)
    
    learned_dag = learn_ges_discrete(
        samples=asia_samples
    )
    
    scores = evaluate_graph_recovery(
        estimated_model=learned_dag,
        true_model=true_model,
        algorithm_name="GES",
        print_scores=True
    )
    
    save_graphviz_dag(
    model_or_graph=learned_dag,
    output_path="results/graphs/asia_ges_learned.png",
    layout="dot",
)
    
    print("\n All scores:")
    print(scores)
    
if __name__ == "__main__":
    main()
    
    
