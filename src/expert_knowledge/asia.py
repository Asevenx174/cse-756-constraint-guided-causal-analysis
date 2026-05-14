from pgmpy.causal_discovery import ExpertKnowledge


def get_asia_expert_knowledge(
    use_expert_knowledge=False,
    knowledge_type="basic",
):
    if not use_expert_knowledge:
        return None

    if knowledge_type == "basic":
        return ExpertKnowledge(
            required_edges = [
        ("asia", "tub")])
        
    if knowledge_type == "ideal":
        return ExpertKnowledge(
            required_edges = [
        ("asia", "tub"),
        ("smoke", "lung"),
        ("smoke", "bronc"),
        ("tub", "either"),
        ("lung", "either"),
        ("either", "xray"),
        ("either", "dysp"),
        ("bronc", "dysp"),
    ],

    forbidden_edges = [
        ("tub", "asia"),
        ("lung", "smoke"),
        ("bronc", "smoke"),
        ("either", "tub"),
        ("either", "lung"),
        ("xray", "either"),
        ("dysp", "either"),
        ("dysp", "bronc"),
    ],

            temporal_order = [
        ["asia", "smoke"],
        ["tub", "lung", "bronc"],
        ["either"],
        ["xray", "dysp"],
    ]
        )
        
    if knowledge_type == "temporal":
        return ExpertKnowledge(
            temporal_order = [
                ["asia", "smoke"],
                ["tub", "lung", "bronc"],
                ["either"],
                ["xray", "dysp"],
    ]
        )

    if knowledge_type == "forbidden":
        return ExpertKnowledge(
            forbidden_edges=[
                ("lung", "smoke")
            ],
        )

    if knowledge_type == "required":
        return ExpertKnowledge(
            required_edges=[
                ("asia", "tub")
            ],
        )

    raise ValueError(
        "knowledge_type must be one of: 'basic', 'forbidden', or 'required' or 'ideal' or 'temporal'."
    )