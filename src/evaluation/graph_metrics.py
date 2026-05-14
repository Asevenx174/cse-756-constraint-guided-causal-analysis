import numpy as np
import networkx as nx
from sklearn.metrics import f1_score

def print_missing_and_extra_edges(
    estimated_model,
    true_model,
    model_name=None,
):
    true_edges = set(true_model.edges())
    estimated_edges = set(estimated_model.edges())

    missing_edges = sorted(true_edges - estimated_edges)
    extra_edges = sorted(estimated_edges - true_edges)

    if model_name is not None:
        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

    print("\nNumber of missing edges:", len(missing_edges))
    print("Missing edges:")
    if missing_edges:
        for edge in missing_edges:
            print(edge)
    else:
        print("None")

    print("\nNumber of extra edges:", len(extra_edges))
    print("Extra edges:")
    if extra_edges:
        for edge in extra_edges:
            print(edge)
    else:
        print("None")
        
def evaluate_graph_recovery(
    estimated_model,
    true_model,
    algorithm_name=None,
    print_scores=True,
):
    """
    Evaluate the learned model skeleton using F1-score.

    This ignores edge direction and compares only whether the correct
    node pairs are connected.
    """
    nodes = list(true_model.nodes())

    estimated_adj = nx.to_numpy_array(
        estimated_model.to_undirected(),
        nodelist=nodes,
        weight=None,
    )

    true_adj = nx.to_numpy_array(
        true_model.to_undirected(),
        nodelist=nodes,
        weight=None,
    )

    f1 = f1_score(
        np.ravel(true_adj),
        np.ravel(estimated_adj),
        zero_division=0,
    )

    if print_scores:
        if algorithm_name is not None:
            print("\nAlgorithm:", algorithm_name)
            print("-" * (len("Algorithm: ") + len(algorithm_name)))

        print("F1-score for the model skeleton:", f1)

    return {
        "skeleton_f1": f1,
        "algorithm": algorithm_name,
    }