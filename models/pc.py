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
