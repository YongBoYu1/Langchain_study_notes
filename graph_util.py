from IPython.display import Image

def visulize_graph(graph_model):
    """Generate a visual representation of the graph model

    Args:
        graph_model (langgraph): a compiled graph model

    Returns:
        IMG: The visual representation of the graph model
    """
    graph_img = Image(graph_model.get_graph().draw_png())
    return graph_img