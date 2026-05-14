from pathlib import Path
from src.simulation.asia import load_asia_model
from src.visualization.graphviz_plot import save_graphviz_dag


def main():
    ground_truth_dag = load_asia_model()

    output_path = "results/graphs/asia_ground_truth.png"

    Path("results/graphs").mkdir(parents=True, exist_ok=True)

    print("\nGround Truth ASIA DAG")
    print("---------------------")
    print("Nodes:")
    print(list(ground_truth_dag.nodes()))

    print("\nEdges:")
    print(list(ground_truth_dag.edges()))

    saved_path = save_graphviz_dag(
        model_or_graph=ground_truth_dag,
        output_path=output_path,
        layout="dot",
    )

    print("\nGround truth DAG saved to:")
    print(saved_path)


if __name__ == "__main__":
    main()