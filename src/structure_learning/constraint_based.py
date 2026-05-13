from pgmpy.causal_discovery import PC

def learn_pc_discrete(
     samples,
     variant="stable",
     ci_test="chi_square",
     return_type="dag",
     significance_level=0.01,
     max_cond_vars=4,
     expert_knowledge=None,
     enforce_expert_knowledge=False,
     n_jobs=-1,
     show_progress=True   
):
    
    estimator = PC(
        variant=variant,
        ci_test=ci_test,
        return_type=return_type,
        significance_level=significance_level,
        max_cond_vars=max_cond_vars,
        expert_knowledge=expert_knowledge,
        enforce_expert_knowledge=enforce_expert_knowledge,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    estimator.fit(samples)

    return estimator.causal_graph_