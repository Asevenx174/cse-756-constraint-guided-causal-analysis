from pgmpy.inference import VariableElimination


def query_probability(
    fitted_model,
    query_variables,
    evidence=None,
    joint=True,
    show_progress=False,
):
    """
    Compute P(query_variables | evidence) using Variable Elimination.

    joint=True:
        returns one joint distribution over all query variables.

    joint=False:
        returns separate marginal distributions for each query variable.
    """
    infer = VariableElimination(fitted_model)

    result = infer.query(
        variables=query_variables,
        evidence=evidence,
        joint=joint,
        show_progress=show_progress,
    )

    return result


def query_map_assignment(
    fitted_model,
    query_variables,
    evidence=None,
    show_progress=False,
):
    """
    Compute the most likely assignment of query_variables given evidence.

    Example:
        Most likely values of lung and bronc given dysp=yes.
    """
    infer = VariableElimination(fitted_model)

    result = infer.map_query(
        variables=query_variables,
        evidence=evidence,
        show_progress=show_progress,
    )

    return result