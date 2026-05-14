import os
from pathlib import Path


def add_graphviz_dll_path():
    """
    Add Graphviz bin directory so pygraphviz can load Graphviz DLLs on Windows.
    """
    if os.name != "nt":
        return

    graphviz_bin_paths = [
        Path(r"C:\Program Files\Graphviz\bin"),
        Path(r"C:\Program Files (x86)\Graphviz\bin"),
    ]

    for graphviz_bin_path in graphviz_bin_paths:
        if graphviz_bin_path.exists():
            os.add_dll_directory(str(graphviz_bin_path))
            os.environ["PATH"] = (
                str(graphviz_bin_path)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
            return

    raise FileNotFoundError(
        "Graphviz bin directory not found. "
        "Please check your Graphviz installation path."
    )


def get_graphviz_dag(model_or_graph, layout="dot"):
    add_graphviz_dll_path()

    dag = model_or_graph.to_graphviz()
    dag.layout(prog=layout)

    return dag


def save_graphviz_dag(model_or_graph, output_path, layout="dot"):
    add_graphviz_dll_path()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dag = get_graphviz_dag(
        model_or_graph=model_or_graph,
        layout=layout,
    )

    dag.draw(str(output_path))

    return str(output_path)