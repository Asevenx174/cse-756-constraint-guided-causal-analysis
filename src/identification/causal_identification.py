from pgmpy.base import DAG
from pgmpy.identification import Adjustment


def convert_to_causal_dag(model_or_graph, exposure, outcome):
    """
    Convert a pgmpy model/graph into a causal DAG with exposure and outcome roles.
    """
    return DAG(
        ebunch=list(model_or_graph.edges()),
        roles={
            "exposures": exposure,
            "outcomes": outcome,
        },
    )


def identify_minimal_adjustment_set(model_or_graph, exposure, outcome):
    """
    Identify one minimal valid adjustment set for the causal effect of exposure on outcome.

    Returns:
        adjustment_set, success, error_message
    """
    dag = convert_to_causal_dag(
        model_or_graph=model_or_graph,
        exposure=exposure,
        outcome=outcome,
    )

    try:
        identified_graph, success = Adjustment(
            variant="minimal"
        ).identify(dag)

        if not success:
            return None, False, "No valid minimal adjustment set was identified."

        adjustment_set = identified_graph.get_role("adjustment")

        if adjustment_set is None:
            adjustment_set = []

        adjustment_set = sorted(list(adjustment_set))

        return adjustment_set, True, None

    except Exception as error:
        return None, False, str(error)


def compare_adjustment_sets(
    true_model,
    learned_dag,
    exposure,
    outcome,
):
    """
    Compare minimal adjustment sets from the true DAG and learned/fitted DAG.
    """
    true_adjustment_set, true_success, true_error = identify_minimal_adjustment_set(
        model_or_graph=true_model,
        exposure=exposure,
        outcome=outcome,
    )

    learned_adjustment_set, learned_success, learned_error = identify_minimal_adjustment_set(
        model_or_graph=learned_dag,
        exposure=exposure,
        outcome=outcome,
    )

    true_set = set(true_adjustment_set) if true_adjustment_set else set()
    learned_set = set(learned_adjustment_set) if learned_adjustment_set else set()

    comparison = {
        "exposure": exposure,
        "outcome": outcome,

        "true_adjustment_set": true_adjustment_set,
        "true_success": true_success,
        "true_error": true_error,

        "learned_adjustment_set": learned_adjustment_set,
        "learned_success": learned_success,
        "learned_error": learned_error,

        "same_adjustment_set": (
            true_success
            and learned_success
            and true_set == learned_set
        ),

        "missing_from_learned": sorted(true_set - learned_set),
        "extra_in_learned": sorted(learned_set - true_set),
    }

    return comparison