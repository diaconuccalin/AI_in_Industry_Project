from data.csuite.csuite_datasets import *
from data.sachs.sachs_datasets import unaltered_dataset
from evaluation.metrics import eval_all
#from models.bnlearn_models import bnlearn_sachs, bnlearn_csuite
#from models.causica_deci import causica_deci
from utils.params import *
from models.gae_model import GAE
import numpy as np
from sklearn.model_selection import ParameterGrid
from plot_network import viz_graph
import sklearn.feature_selection
from castle.common import GraphDAG
from castle.metrics import MetricsDAG



# Adapted from https://www.bnlearn.com/research/sachs05/
# and https://github.com/vspinu/bnlearn/blob/fec3cc0371fabfb6fe7abe0574038522beb2aa13/R/discretize.R
# and https://dspace.mit.edu/handle/1721.1/8699?show=full
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


def seed_everything():
    torch.manual_seed(SEED)

    return SEED


def main_sanity_check():
    available_csuite_datasets = [
        lingauss, linexp, nonlingauss, nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
    ]

    for csuite_dataset in available_csuite_datasets:
        sanity_check(csuite_dataset)

    return None


# Paper: https://arxiv.org/pdf/1911.07420
# Inspired by: https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle

# According to the paper h(A) = trace(e^[A°A]) - d.
# Where e^M denotes the matrix exponential of a matrix M and ° denotes the Hadamard product.
#This is basically a smooth constraint on A.
def compute_h(w_adj):

    d = w_adj.shape[0]
    h = torch.trace(torch.matrix_exp(w_adj * w_adj)) - d

    return h


def train_gae(model, x, n, lr, alpha, beta, rho, l1_penalty, rho_threshold, h_threshold, gamma, early_stopping=False, update_freq=2000):
    #seed_everything()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    h = np.inf
    prev_mse = np.inf
    adj_matrix = None
    prev_adj_matrix = None
    epochs = 10
    init_iter = 2
    early_stopping_thresh=1.0
    
    # alpha is the lagrangian multiplier, rho is the penalty parameter, beta and gamma are tuning parameters

    for epoch in range(epochs):
        #print(epoch)
        
        while rho < rho_threshold:

            for _ in range(update_freq):
                
                optimizer.zero_grad()

                curr_mse, curr_w_adj = model(x)
                curr_h = compute_h(curr_w_adj)
                
                # augmented lagrangian, look at the formula on the paper
                loss = ((0.5 / n) * curr_mse + l1_penalty * torch.norm(curr_w_adj, p=1) + alpha * curr_h + 0.5 * rho * curr_h ** 2)

                loss.backward()
                optimizer.step()
                
            # Update rule for Rho, rho = rho * beta if h(A^k+1) >=  gamma * h(A^k)
            if curr_h > gamma * h:
                    rho *= beta
            else:
                    break
            
        if early_stopping:
            if (curr_mse / prev_mse > early_stopping_thresh and curr_h <= 1e-7):
                return prev_adj_matrix
            else:
                prev_adj_matrix = curr_w_adj
                prev_mse = curr_mse
            
        adj_matrix, h = curr_w_adj, curr_h
        
        # Update rule for alpha, alpha = alpha + rho * h(A^k+1) 
        alpha += rho * curr_h.detach().cpu()

        if h <= h_threshold and epoch > init_iter:
            break

    return adj_matrix


def prepare_input_data(x):
    
    n, d = x.shape[:2]

    if len(x.shape) == 2:
        x = x.reshape(n, d, 1)
        input_dim = x.shape[2]
    elif len(x.shape) == 3:
        input_dim = x.shape[2]
    
    #data_loader = DataLoader(x, batch_size, shuffle=True)

    return n, d, x, input_dim
    
    
def gae_tuning():
    
    seed_everything()
    
    best_results = {}
    
    param_grid = {
       'beta': [2, 2.5, 3, 3.5, 4],
       'gamma': [0.10, 0.25, 0.35, 0.45],
       'hidden_dim':[4, 8, 16, 32]
    }
    
    graph_threshold = 0.3
    
    available_csuite_datasets = [
        nonlin_simpson, symprod_simpson, large_backdoor, weak_arrows
    ]
  
    for csuite_dataset in available_csuite_datasets:
        df, gt_graph = csuite_dataset(500, True, True)
        
        best_adj_score = 0  # Initialize best combined score for this dataset
        best_orientation_score = 0
        best_params_adj = None  # Initialize best parameters for this dataset
        best_params_orientation = None # Initialize best parameters for this dataset
        

        df = df.to_numpy()
        
        x = torch.from_numpy(df)

        for params in ParameterGrid(param_grid):
            adj_score = 0
            orientation_score = 0
            
            beta = params['beta']
            gamma = params['gamma']
            hidden_dim = params['hidden_dim']
            
            n, d, x, input_dim = prepare_input_data(x)
        
            model = GAE(d, input_dim, latent_dim=1 ,hidden_dim=hidden_dim)
        
            adj_matrix = train_gae(model, x, n, lr=1e-3, alpha=0.0, beta=beta, rho=1.0, 
                               l1_penalty=0.0, 
                               rho_threshold=1e30,
                               h_threshold=1e-8,
                               gamma=gamma,
                               early_stopping=True,
                               update_freq=200
                            )
        
            # Normalize the adjacency matrix
            adj_matrix = (adj_matrix / torch.max(abs(adj_matrix)))
            adj_matrix = adj_matrix.detach().cpu().numpy()
        
            causal_matrix = (np.abs(adj_matrix) > graph_threshold).astype(int)
        
            #print(csuite_dataset.__name__)
            f1_scores = (eval_all(torch.tensor(causal_matrix), gt_graph))
            
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
        print(f"Best results for {csuite_dataset.__name__}:")

    return best_results

        
        

def gae_sachs(apply_hartemink_discretization=False, beta = 2, gamma = 0.25, hidden_dim = 32, update_freq = 2000):
    seed_everything()
    graph_threshold = 0.3
    
    df, correlation_dict, gt_adj_graph = unaltered_dataset(
        get_data=True,
        return_index_name_correlation=True,
        return_adjacency_graph=True)
    
    if apply_hartemink_discretization:
        df = hartemink_discretization(df)
        
    df = df.to_numpy()
        
    x = torch.from_numpy(df)
        
    n, d, x, input_dim = prepare_input_data(x)
        
    model = GAE(d, input_dim, latent_dim=1 ,hidden_dim=hidden_dim)
        
    adj_matrix = train_gae(model, x, n, lr=1e-3, alpha=0.0, beta=beta, rho=1.0, 
                               l1_penalty=0.0, 
                               rho_threshold=1e30,
                               h_threshold=1e-8,
                               gamma=gamma,
                               update_freq=update_freq
                            )
        
    # Normalize the adjacency matrix
    adj_matrix = (adj_matrix / torch.max(abs(adj_matrix)))
    adj_matrix = adj_matrix.detach().cpu().numpy()
        
    causal_matrix = (np.abs(adj_matrix) > graph_threshold).astype(int)
        
    #print("Sachs")
    f1_scores = (eval_all(torch.tensor(causal_matrix), gt_adj_graph))
    print(f1_scores)
        
    fig = viz_graph(causal_matrix, gt_adj_graph.numpy(), title1="Sachs", title2="Sachs_GT")
    GraphDAG(causal_matrix, gt_adj_graph.numpy(), 'result')
    mt = MetricsDAG(causal_matrix, gt_adj_graph.numpy())
    #print(mt.metrics)
    return causal_matrix, gt_adj_graph.numpy(), f1_scores, mt



def gae_cs(dataset, apply_hartemink_discretization=False, beta = 2, gamma = 0.25, hidden_dim = 32, samples = 2000, update_freq = 2000):
    seed_everything()
    
    graph_threshold = 0.3
        
    df, gt_graph = dataset(samples, True, True)
        
    if apply_hartemink_discretization:
         df = hartemink_discretization(df)

    df = df.to_numpy()
        
    x = torch.from_numpy(df)
        
    n, d, x, input_dim = prepare_input_data(x)
        
    model = GAE(d, input_dim, latent_dim=1 ,hidden_dim = hidden_dim)
        
    adj_matrix = train_gae(model, x, n, lr=1e-3, alpha=0.0, beta=beta, rho=1.0, 
                               l1_penalty=0.0, 
                               rho_threshold=1e30,
                               h_threshold=1e-8,
                               gamma=gamma,
                               update_freq=update_freq
                            )
        
    # Normalize the adjacency matrix
    adj_matrix = (adj_matrix / torch.max(abs(adj_matrix)))
    adj_matrix = adj_matrix.detach().cpu().numpy()
        
    causal_matrix = (np.abs(adj_matrix) > graph_threshold).astype(int)
        
    #print(dataset.__name__)
    f1_scores = (eval_all(torch.tensor(causal_matrix), gt_graph))
    print(f1_scores)
        
    fig = viz_graph(causal_matrix, gt_graph.numpy(), title1=dataset.__name__, title2=dataset.__name__ + "GT")
    GraphDAG(causal_matrix, gt_graph.numpy(), 'result')
    mt = MetricsDAG(causal_matrix, gt_graph.numpy())
    #print(mt.metrics)
    return causal_matrix, gt_graph.numpy(), f1_scores, mt
    
   

def sachs_tuning_gae():
    best_results = {}
    graph_threshold = 0.3
    
    param_grid = {
       'beta': [2, 2.5, 3, 3.5, 4],
       'gamma': [0.10, 0.25, 0.35, 0.45],
       'hidden_dim':[4, 8, 16, 32]
    }

    df, correlation_dict, gt_adj_graph = unaltered_dataset(
        get_data=True,
        return_index_name_correlation=True,
        return_adjacency_graph=True)
    
    df = df.to_numpy()
    x = torch.from_numpy(df)
        
    best_adj_score = 0  
    best_orientation_score = 0
    best_params_adj = None  
    best_params_orientation = None 
        

    for params in ParameterGrid(param_grid):
            adj_score = 0
            orientation_score = 0
            
            beta = params['beta']
            gamma = params['gamma']
            hidden_dim = params['hidden_dim']
            
            n, d, x, input_dim = prepare_input_data(x)
        
            model = GAE(d, input_dim, latent_dim=1 ,hidden_dim=hidden_dim)
        
            adj_matrix = train_gae(model, x, n, lr=1e-3, alpha=0.0, beta=beta, rho=1.0, 
                               l1_penalty=0.0, 
                               rho_threshold=1e30,
                               h_threshold=1e-8,
                               gamma=gamma,
                               update_freq=200
                            )
            
            # Normalize the adjacency matrix
            adj_matrix = (adj_matrix / torch.max(abs(adj_matrix)))
            adj_matrix = adj_matrix.detach().cpu().numpy()
        
            causal_matrix = (np.abs(adj_matrix) > graph_threshold).astype(int)
    
            f1_scores = (eval_all(torch.tensor(causal_matrix), gt_adj_graph))
            
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
    best_results['sachs'] = {'best_adj_score': best_adj_score, 'best_orientation_score' : best_orientation_score ,'best_params_adj': best_params_adj, 'best_params_or': best_params_orientation}
    print("The tuning of Sachs is done.")
    return best_results