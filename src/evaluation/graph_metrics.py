import numpy as np
import networkx as nx
from sklearn.metrics import f1_score


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