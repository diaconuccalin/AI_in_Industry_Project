import numpy as np
import pandas as pd
import sklearn.feature_selection
import torch
from pybnesian import hc, GaussianNetworkType

from data.csuite.csuite_datasets import lingauss
from data.sachs.sachs_datasets import unaltered_dataset
from evaluation.metrics import eval_all
from utils.params import SEED


# Adapted from https://www.bnlearn.com/research/sachs05/
# and https://github.com/vspinu/bnlearn/blob/fec3cc0371fabfb6fe7abe0574038522beb2aa13/R/discretize.R
# and https://dspace.mit.edu/handle/1721.1/8699?show=full
def hartemink_discretization(df, breaks=3, initial_breaks=60):
    # Cache useful quantities
    nodes = df.columns

    # Perform initial discretization
    quantile_df = df.copy(deep=True)
    for col in nodes:
        quantile_df[col] = pd.qcut(list(df[col]), initial_breaks, labels=False)

    # Count down through discretization levels
    for n_levels in range(initial_breaks, breaks, -1):
        if n_levels % 10 == 0:
            print(f"Working on the {n_levels}th level of discretization.")
        # Go through each variable
        for node in nodes:
            # Prepare df by isolating current node
            x = quantile_df.loc[:, quantile_df.columns != node]
            y = quantile_df[node]
            mutual_information_values = list()

            # Go through level pairs
            for collapsing in range(1, n_levels):
                local_y = np.copy(y)
                local_y[local_y >= collapsing] -= 1

                # Compute mutual information
                mutual_information_values.append(sum(sklearn.feature_selection.mutual_info_regression(x, local_y)))

            # Collapse according to best result
            quantile_df[node][y >= (np.argmax(mutual_information_values) + 1)] -= 1

    return quantile_df.astype("float64")


def bnlearn_solution(df, apply_hartemink_discretization=True):
    if apply_hartemink_discretization:
        df = hartemink_discretization(df)

    # Apply hill climbing greedy search with bge score
    learned = hc(
        df=df,
        bn_type=GaussianNetworkType(),
        score="bge",
        seed=SEED
    )

    return learned


def bnlearn_sachs(apply_hartemink_discretization=False):
    # Get sachs set
    df, correlation_dict, gt_adj_graph = unaltered_dataset(
        get_data=True,
        return_index_name_correlation=True,
        return_adjacency_graph=True)

    # Apply bnlearn model
    learned = bnlearn_solution(df, apply_hartemink_discretization=apply_hartemink_discretization)

    # Construct adjacency graph
    pred_adj_graph = np.zeros_like(np.array(gt_adj_graph))
    for origin, destination in learned.arcs():
        pred_adj_graph[
            correlation_dict[origin],
            correlation_dict[destination]
        ] = 1

    # Print scores
    print(eval_all(torch.Tensor(pred_adj_graph), gt_adj_graph))


def bnlearn_csuite(csuite_dataset=lingauss, samples=500, apply_hartemink_discretization=False, estimate_cpds=False):
    # Get set
    df, gt_adj_graph = csuite_dataset(
        samples,
        generate_data=True,
        return_adjacency_graph=True
    )

    # Apply bnlearn model
    learned = bnlearn_solution(df, apply_hartemink_discretization=apply_hartemink_discretization)

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
