# Paper: https://arxiv.org/pdf/1911.07420
# Inspired by: https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle

import torch
import torch.nn as nn

class GAE(nn.Module):
    def __init__(self, d, input_dim, latent_dim, hidden_dim):
        super(GAE, self).__init__()
        self.d = d
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            #first layer [input_dim, hidden_dim]
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            
            #second layer[hidden_dim, hidden_dim]
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            
            #[hidden_dim, latent_dim]
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            #first layer [latent_dim, hidden_dim]
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            
            #second layer [hidden_dim, hidden_dim]
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            
            #[hidden_dim, input_dim]
            nn.Linear(hidden_dim, input_dim)
            
        )
        
        # We create a weight matrix with random values between -0.1 and 0.1
        w = torch.nn.init.uniform_(torch.empty(self.d, self.d,), a=-0.1, b=0.1)
        
        # And then convert w into a trainable parameter of the model
        self.w = torch.nn.Parameter(w)
        
        self.mseLoss = nn.MSELoss()
        self.double()

    
    def forward(self, x):
        
        # We subtract the identity matrix from the value 1 to invert the diagonal elements of the identity matrix, making them 0 while the other elements remain 1.
        # Then perform element-wise multiplication with the original adjacency matrix w_adj. 
        # By doing this we set all diagonal elements of the adjacency matrix to 0, while preserving the off-diagonal elements that represent connections between nodes in the graph.
        self.w_adj = (1. - torch.eye(self.w.shape[0])) * self.w
  
        latent_output = self.encoder(x)
        
        # Compute A^t * H(output of the encoder) and then get the reconstructed output from the decoder
        output = torch.matmul(self.w_adj.t(), latent_output)
        
        # Reconstructed output
        x_est = self.decoder(output)
        
        reconstruct_loss = self.mseLoss(x, x_est)

        return reconstruct_loss, self.w_adj


