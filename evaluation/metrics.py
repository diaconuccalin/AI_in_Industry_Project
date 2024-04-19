from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1


def eval_all(pred_graph, gt_graph):
    """
    Evaluate a predicted graph against a given one, using causica metrics.

    :param pred_graph: torch.tensor - adjacency matrix of the predicted graph (with values of 0 or 1)
    :param gt_graph: torch.tensor - adjacency matrix of the ground truth graph (with values of 0 or 1)
    :return: dictionary with 2 elements that have as keys "adjacency_f1" and "orientation_f1, and floats as values
    """

    return {
        "adjacency_f1": adjacency_f1(pred_graph, gt_graph).item(),
        "orientation_f1": orientation_f1(pred_graph, gt_graph).item()
    }
