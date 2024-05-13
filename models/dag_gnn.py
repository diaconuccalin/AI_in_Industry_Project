import numpy as np
from torch import *
from data.csuite.csuite_datasets import *
from evaluation.metrics import eval_all
from castle.algorithms import DAG_GNN
from data.sachs.sachs_datasets import unaltered_dataset
from plot_network import viz_graph
from sklearn.model_selection import ParameterGrid
from models.gae_main import hartemink_discretization
from castle.common import GraphDAG
from castle.metrics import MetricsDAG


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def prepare_input_data(x):
    
    n, d = x.shape[:2]

    if len(x.shape) == 2:
        x = x.reshape(n, d, 1)
        input_dim = x.shape[2]
    elif len(x.shape) == 3:
        input_dim = x.shape[2]
    
    #data_loader = DataLoader(x, batch_size, shuffle=True)

    return d, x, input_dim
    


def gnn_tuning(samples = 500):
    best_results = {}
    
    device = get_default_device()
    
    param_grid = {
       'eta': [4, 6, 8, 10, 12],
       #'batch_size': [16, 32, 64],
       'gamma': [0.10, 0.15, 0.25, 0.35, 0.45],
       'hidden_dim':[4, 8, 16, 32]
    }
    
    available_csuite_datasets = [
        lingauss, linexp, nonlingauss, nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
    ]
  
    for csuite_dataset in available_csuite_datasets:
        df, gt_graph = csuite_dataset(samples, True, True)
        
        best_adj_score = 0  
        best_orientation_score = 0
        best_params_adj = None  
        best_params_orientation = None 
        

        df = df.to_numpy()
        
        x = torch.from_numpy(df)

        for params in ParameterGrid(param_grid):
            adj_score = 0
            orientation_score = 0
            
            eta = params['eta']
            gamma = params['gamma']
            hidden_dim = params['hidden_dim']
            
            dag_gnn = DAG_GNN(epochs=10, encoder_hidden=hidden_dim, decoder_hidden=hidden_dim ,eta= eta, gamma= gamma, device_type=device)
            dag_gnn.learn(x)
    
            f1_scores = (eval_all(torch.tensor(dag_gnn.causal_matrix), gt_graph))
            
            # Calculate combined score 
            adj_score = f1_scores['adjacency_f1'] 
            orientation_score = f1_scores['orientation_f1']

            # Check if current combined score is better than previous best
            if adj_score > best_adj_score:
                best_adj_score = adj_score
                best_params_adj = params
                
            if orientation_score > best_orientation_score:
                best_orientation_score = orientation_score
                best_params_orientation = params
                

        # Update best results for this dataset
        best_results[csuite_dataset.__name__] = {'best_adj_score': best_adj_score, 'best_orientation_score' : best_orientation_score ,'best_params_adj': best_params_adj, 'best_params_or': best_params_orientation}
        print(f"The tuning of {csuite_dataset.__name__} is done.")
    return best_results



    
def dag_gnn_cs(dataset, apply_hartemink_discretization=False, hidden_dim=64, eta=10, gamma=1, samples = 2000):
    
    device = get_default_device()
    
    df, gt_graph = dataset(samples, True, True)
    
    if apply_hartemink_discretization:
        df = hartemink_discretization(df)

    df = df.to_numpy()
    x = torch.from_numpy(df)
       
    dag_gnn = DAG_GNN(encoder_hidden=hidden_dim, decoder_hidden=hidden_dim ,eta= eta, gamma= gamma, device_type=device)
    dag_gnn.learn(x)
    
    f1_scores = (eval_all(torch.tensor(dag_gnn.causal_matrix), gt_graph))
    print(f1_scores)
     
    fig = viz_graph(dag_gnn.causal_matrix, gt_graph.numpy(), title1=dataset.__name__, title2=dataset.__name__ + "_GT")
    GraphDAG(dag_gnn.causal_matrix, gt_graph.numpy(), 'result')
    mt = MetricsDAG(dag_gnn.causal_matrix, gt_graph.numpy())
    #print(mt.metrics)
    return dag_gnn.causal_matrix, gt_graph.numpy(), f1_scores, mt



   
def dag_gnn_sachs(apply_hartemink_discretization=False, hidden_dim=64, eta=10, gamma=1):
    device = get_default_device()
    
    df, correlation_dict, gt_adj_graph = unaltered_dataset(
        get_data=True,
        return_index_name_correlation=True,
        return_adjacency_graph=True)
    
    if apply_hartemink_discretization:
        df = hartemink_discretization(df)

    df = df.to_numpy()
    x = torch.from_numpy(df)
       
    dag_gnn = DAG_GNN(encoder_hidden=hidden_dim, decoder_hidden=hidden_dim ,eta= eta, gamma= gamma,device_type=device)
    dag_gnn.learn(x)
    
    f1_scores = (eval_all(torch.tensor(dag_gnn.causal_matrix), gt_adj_graph))
    print(f1_scores)
     
    fig = viz_graph(dag_gnn.causal_matrix, gt_adj_graph.numpy(), title1="Sachs", title2="Sachs_GT")
    GraphDAG(dag_gnn.causal_matrix, gt_adj_graph.numpy(), 'result')
    mt = MetricsDAG(dag_gnn.causal_matrix, gt_adj_graph.numpy())
    #print(mt.metrics)
    return dag_gnn.causal_matrix, gt_adj_graph.numpy(), f1_scores, mt

  
