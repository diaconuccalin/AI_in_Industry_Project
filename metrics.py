from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1


def eval_all(pred_graph, gt_graph):
    return {
        "adjacency_f1": adjacency_f1(pred_graph, gt_graph).item(),
        "orientation_f1": orientation_f1(pred_graph, gt_graph).item()
    }
