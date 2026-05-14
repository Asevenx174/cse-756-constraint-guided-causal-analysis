from pgmpy.inference import VariableElimination


def query_probability(
    fitted_model,
    query_variables,
    evidence=None,
    show_progress=False,
):
    """
    Compute posterior probability P(query_variables | evidence).

    fitted_model:
        Bayesian network with learned CPDs.

    query_variables:
        List of variables we want to infer.

    evidence:
        Observed variable values.
    """
    infer = VariableElimination(fitted_model)

    result = infer.query(
        variables=query_variables,
        evidence=evidence,
        show_progress=show_progress,
    )

    return result