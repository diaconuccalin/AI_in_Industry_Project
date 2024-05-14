import torch
import pandas as pd
import numpy as np
from data.csuite.csuite_datasets import lingauss,linexp, nonlingauss, nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
from data.sachs.sachs_datasets import unaltered_dataset
from evaluation.metrics import eval_all
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import RL, PC

def hartemink_discretization(df=None, breaks=3, initial_breaks=60):
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

def pc(df, gt_adj_graph, method_name):

  pc = PC(variant='original')
  pc.learn(df.values)

  true_dag = np.array(gt_adj_graph)
  
  GraphDAG(pc.causal_matrix, true_dag)
   
  met = MetricsDAG(pc.causal_matrix, true_dag)
  print(met.metrics)

  met2 = eval_all(torch.Tensor(pc.causal_matrix), gt_adj_graph)

  print(method_name, ": ", met2)

  return met2
