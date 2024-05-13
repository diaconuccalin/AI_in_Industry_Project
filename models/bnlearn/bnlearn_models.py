import numpy as np
import torch
from pybnesian import hc, GaussianNetworkType

from data.csuite.csuite_datasets import lingauss
from data.sachs.sachs_datasets import unaltered_dataset
from evaluation.metrics import eval_all
from utils.params import SEED
from utils.solution_utils import hartemink_discretization


# Adapted from https://www.bnlearn.com/research/sachs05/
# and https://github.com/vspinu/bnlearn/blob/fec3cc0371fabfb6fe7abe0574038522beb2aa13/R/discretize.R
# and https://dspace.mit.edu/handle/1721.1/8699?show=full
def bnlearn_solution(df, apply_hartemink_discretization=True, score="bge"):
    if apply_hartemink_discretization:
        df = hartemink_discretization(df)

    # Apply hill climbing greedy search with bge score
    learned = hc(
        df=df,
        bn_type=GaussianNetworkType(),
        score=score,
        seed=SEED
    )

    return learned


def bnlearn_sachs(apply_hartemink_discretization=False, score="bge"):
    # Get sachs set
    df, correlation_dict, gt_adj_graph = unaltered_dataset(
        get_data=True,
        return_index_name_correlation=True,
        return_adjacency_graph=True)

    # Apply bnlearn model
    learned = bnlearn_solution(df, apply_hartemink_discretization=apply_hartemink_discretization, score=score)

    # Construct adjacency graph
    pred_adj_graph = np.zeros_like(np.array(gt_adj_graph))
    for origin, destination in learned.arcs():
        pred_adj_graph[
            correlation_dict[origin],
            correlation_dict[destination]
        ] = 1

    # Print scores
    print(eval_all(torch.Tensor(pred_adj_graph), gt_adj_graph))


def bnlearn_csuite(
        csuite_dataset=lingauss,
        samples=500,
        apply_hartemink_discretization=False,
        estimate_cpds=False,
        score="bge"
):
    # Get set
    df, gt_adj_graph = csuite_dataset(
        samples,
        generate_data=True,
        return_adjacency_graph=True
    )

    # Apply bnlearn model
    learned = bnlearn_solution(df, apply_hartemink_discretization=apply_hartemink_discretization, score=score)

    # Construct adjacency graph
    pred_adj_graph = np.zeros_like(np.array(gt_adj_graph))
    for origin, destination in learned.arcs():
        pred_adj_graph[
            int(origin[2:]),
            int(destination[2:])
        ] = 1

    # Print scores
    print("\n~~~~~~~~~~ RESULTS ~~~~~~~~~~")
    print(eval_all(torch.Tensor(pred_adj_graph), gt_adj_graph))
    print()

    # Estimate cpds
    if estimate_cpds:
        print("\n~~~~~~~~~~ CPDs ~~~~~~~~~~")
        learned.fit(df)

        for node in learned.nodes():
            print(learned.cpd(node))
