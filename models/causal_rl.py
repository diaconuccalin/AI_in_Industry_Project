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

def causal_rl(df, gt_adj_graph, method_name, params):

  rl = RL(**params)
  rl.learn(df.values)

  # Plot the estimated DAG
  true_dag = np.array(gt_adj_graph)
  
  GraphDAG(rl.causal_matrix, true_dag)
   
  met = MetricsDAG(rl.causal_matrix, true_dag)
  print(met.metrics)

  met2 = eval_all(torch.Tensor(rl.causal_matrix), gt_adj_graph)

  print(method_name, ": ", met2)

  return met2

def train_evaluate_model(model, df, gt_adj_graph , hyperparameters):
    rl_model = model(**hyperparameters)
    rl_model.learn(df)

    met = eval_all(torch.Tensor(rl_model.causal_matrix), true_dag)
    rounded_met = {key: round(value, 2) for key, value in met.items()}

    return met
