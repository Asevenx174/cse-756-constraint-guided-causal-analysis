from pgmpy.causal_discovery import HillClimbSearch, GES

def learn_hill_climb_discrete(
    samples,
    scoring_method="bic-d",
    start_dag=None,
    tabu_length=100,
    max_indegree=None,
    expert_knowledge=None,
    return_type="dag",
    epsilon=0.0001,
    max_iter=1000000,
    show_progress=True,
    return_estimator=False,
):
     estimator = HillClimbSearch(
        scoring_method=scoring_method,
        start_dag=start_dag,
        tabu_length=tabu_length,
        max_indegree=max_indegree,
        expert_knowledge=expert_knowledge,
        return_type=return_type,
        epsilon=epsilon,
        max_iter=max_iter,
        show_progress=show_progress,
    )

     estimator.fit(samples)

     if return_estimator:
        return estimator

     return estimator.causal_graph_ 
 
 
def learn_ges_discrete(
    samples,
    scoring_method="bic-d",
    return_type="dag",
    min_improvement=1e-6,
    return_estimator=False,
):
    estimator = GES(
        scoring_method=scoring_method,
        return_type=return_type,
        min_improvement=min_improvement,
    )

    estimator.fit(samples)

    if return_estimator:
        return estimator

    return estimator.causal_graph_