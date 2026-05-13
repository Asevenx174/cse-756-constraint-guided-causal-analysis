import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score


def get_skeleton_precision_recall_f1(
    estimated_model,
    true_model,
    print_scores=True,
):
    nodes = list(true_model.nodes())

    true_adj = nx.to_numpy_array(
        true_model.to_undirected(),
        nodelist=nodes,
        weight=None,
    )

    estimated_adj = nx.to_numpy_array(
        estimated_model.to_undirected(),
        nodelist=nodes,
        weight=None,
    )

    y_true = np.ravel(true_adj)
    y_pred = np.ravel(estimated_adj)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "skeleton_precision": precision,
        "skeleton_recall": recall,
        "skeleton_f1": f1,
    }

    if print_scores:
        print("Skeleton precision:", precision)
        print("Skeleton recall:", recall)
        print("Skeleton F1-score:", f1)

    return results


def get_directed_edge_precision_recall_f1(
    estimated_model,
    true_model,
    print_scores=True,
):
    nodes = list(true_model.nodes())

    y_true = []
    y_pred = []

    for source in nodes:
        for target in nodes:
            if source == target:
                continue

            y_true.append(1 if true_model.has_edge(source, target) else 0)
            y_pred.append(1 if estimated_model.has_edge(source, target) else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "directed_precision": precision,
        "directed_recall": recall,
        "directed_f1": f1,
    }

    if print_scores:
        print("Directed-edge precision:", precision)
        print("Directed-edge recall:", recall)
        print("Directed-edge F1-score:", f1)

    return results


def get_orientation_precision_recall_f1(
    estimated_model,
    true_model,
    print_scores=True,
):
    true_skeleton_edges = {
        frozenset((u, v)) for u, v in true_model.to_undirected().edges()
    }

    estimated_skeleton_edges = {
        frozenset((u, v)) for u, v in estimated_model.to_undirected().edges()
    }

    common_skeleton_edges = true_skeleton_edges.intersection(
        estimated_skeleton_edges
    )

    if len(common_skeleton_edges) == 0:
        results = {
            "orientation_precision": 0.0,
            "orientation_recall": 0.0,
            "orientation_f1": 0.0,
            "num_common_skeleton_edges": 0,
        }

        if print_scores:
            print("Orientation precision: 0.0")
            print("Orientation recall: 0.0")
            print("Orientation F1-score: 0.0")
            print("Common skeleton edges: 0")

        return results

    y_true = []
    y_pred = []

    for edge in common_skeleton_edges:
        u, v = tuple(edge)

        # Direction is represented relative to the arbitrary pair order (u, v).
        # If true graph has u -> v, label = 1.
        # If true graph has v -> u, label = 0.
        true_direction = 1 if true_model.has_edge(u, v) else 0
        estimated_direction = 1 if estimated_model.has_edge(u, v) else 0

        y_true.append(true_direction)
        y_pred.append(estimated_direction)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "orientation_precision": precision,
        "orientation_recall": recall,
        "orientation_f1": f1,
        "num_common_skeleton_edges": len(common_skeleton_edges),
    }

    if print_scores:
        print("Orientation precision:", precision)
        print("Orientation recall:", recall)
        print("Orientation F1-score:", f1)
        print("Common skeleton edges:", len(common_skeleton_edges))

    return results


def evaluate_graph_recovery(
    estimated_model,
    true_model,
    algorithm_name=None,
    print_scores=True,
):
    if print_scores and algorithm_name is not None:
        print("\nAlgorithm:", algorithm_name)
        print("-" * (len("Algorithm: ") + len(algorithm_name)))

    skeleton_scores = get_skeleton_precision_recall_f1(
        estimated_model=estimated_model,
        true_model=true_model,
        print_scores=print_scores,
    )

    directed_scores = get_directed_edge_precision_recall_f1(
        estimated_model=estimated_model,
        true_model=true_model,
        print_scores=print_scores,
    )

    orientation_scores = get_orientation_precision_recall_f1(
        estimated_model=estimated_model,
        true_model=true_model,
        print_scores=print_scores,
    )

    results = {
        **skeleton_scores,
        **directed_scores,
        **orientation_scores,
    }

    if algorithm_name is not None:
        results["algorithm"] = algorithm_name

    return results