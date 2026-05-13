def get_graphviz_dag(model_or_graph, layout="dot"):
    dag = model_or_graph.to_graphviz()
    dag.layout(prog=layout)
    return dag

def save_graphviz_dag(model_or_graph, output_path, layout="dot"):
    dag = get_graphviz_dag(model_or_graph, layout=layout)
    dag.draw(output_path)
    return output_path