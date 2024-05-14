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

def train_evaluate_model(df, gt_adj_graph , hyperparameters):
  rl_model = RL(**hyperparameters)
  rl_model.learn(df.values)

  met = eval_all(torch.Tensor(rl_model.causal_matrix), gt_adj_graph)
  rounded_met = {key: round(value, 2) for key, value in met.items()}

  return met

def tuning_rl(param_grid, datasets):
  
  param_combinations = list(ParameterGrid(param_grid))
  # Perform grid search for each dataset using parallel execution
  results_rl = {}
  for dataset_func in datasets:
    df, gt_adj_graph = dataset_func(20, True, True)
    results_rl[dataset_func.__name__] = {}
      # Generate all possible combinations of hyperparameters
  # Print the generated combinations
    for params in param_combinations:
      params_tuple = tuple(params.items())
      results_rl[dataset_func.__name__][params_tuple] = train_evaluate_model(df, gt_adj_graph, params)
  return results_rl
