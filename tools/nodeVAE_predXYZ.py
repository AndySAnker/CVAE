import os, sys, math, pdb, torch, dgl, scipy, time, random
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal, Independent, MultivariateNormal
from torch.distributions.kl import kl_divergence as KLD
from torch.nn.functional import softplus
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from gnnModules import GatedConv1d, GatedConvTranspose1d
torch.manual_seed(12)
random.seed(12)
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

def reconstruction_loss(structures, XYZPreds, config, epoch, dummy_center):
    Val_Loss = 0
    structure_list = []
    for iter, (structure, pred_coord) in enumerate(zip(structures, XYZPreds)):
        df_XYZ_true = pd.read_csv(config['data_path']+"/"+structure+".xyz", skiprows=2, sep=" ", skipinitialspace=1, names=["atom", "x", "y", "z"])
        true_coord = torch.tensor(df_XYZ_true[["x","y","z"]].values, dtype=torch.float) 
        true_coord = torch.cat((true_coord, dummy_center*torch.ones((config['nNodes']-len(true_coord), 3))), axis=0)
        diff = pred_coord - true_coord
       # if epoch == 50:
       #     pdb.set_trace()
        #Val_Loss += torch.abs(diff).mean()#**2).mean()
        Val_Loss += (diff**2).mean()
        structure_list.append(structure)
    return Val_Loss / len(structures)

class Conditioning(nn.Module):
    def __init__(self, numNodes, cond_dim):
        super().__init__()

        self.condPDF = nn.Sequential(
            GatedConvTranspose1d(301,48,kernel_size=1,stride=1), nn.ReLU(), #nn.BatchNorm1d(48),#nn.Dropout(0.1), nn.BatchNorm1d(24),
            GatedConvTranspose1d(48,24,kernel_size=1,stride=1), nn.ReLU(), #nn.BatchNorm1d(24),
            GatedConvTranspose1d(24,numNodes,kernel_size=cond_dim,stride=cond_dim))
        
    def forward(self, y_PDF):
        for conv in self.condPDF:
            y_PDF = conv(y_PDF)
        
        return y_PDF

class Encoder(nn.Module):
    def __init__(self, numNodesF, numNodes, nhid, latent_space, cond_dim):
        super().__init__()

        self.numNodes = numNodes
        self.latent_space = latent_space
        m = nn.ReLU()
        # Encoder GCN
        self.encoder = nn.ModuleList([
            decode_layer(numNodesF+cond_dim+3, nhid * 6, m), #nn.BatchNorm1d(nhid * 6),#nn.Dropout(0.1), #nn.BatchNorm1d(nhid * 6),
            decode_layer(nhid * 6, nhid * 4, m), #nn.BatchNorm1d(nhid * 4),
            decode_layer(nhid * 4, nhid * 2, m), #nn.BatchNorm1d(nhid * 2),
            decode_layer(nhid * 2, latent_space * 2, None)])  # note the 2*nhid
        
        self.Node_MLP = nn.Linear(numNodes, 1)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)

        for conv in self.encoder:
            x = conv(x)  # Graph convolution
        x = self.Node_MLP(x.transpose(1,2)).squeeze()
        emb = x.clone()

        # Split encoder outputs into a mean and variance vector
        mu, log_var = torch.chunk(emb, 2, dim=-1)
        # Make sure that the log variance is positive
        log_var = softplus(log_var)
        sigma = torch.exp(log_var / 2)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, nhid, numNodes, latent_space, satellites):
        super().__init__()
        
        self.numNodes = numNodes
        self.satellites = satellites
        m = nn.ReLU()
        tanh = nn.Tanh()
        sig = nn.Sigmoid()
        # The latent code must be decoded into the original node features
        self.decoder_XYZ = nn.ModuleList([
            decode_layer(latent_space, nhid * 2, m), #nn.BatchNorm1d(nhid * 2),#nn.Dropout(0.1), nn.BatchNorm1d(nhid * 2),
            decode_layer(nhid * 2, nhid * 4, m), #nn.BatchNorm1d(nhid * 4),
            decode_layer(nhid * 4, nhid * 6, m), #nn.BatchNorm1d(nhid * 6),
            decode_layer(nhid * 6, nhid * 8, m), #nn.BatchNorm1d(nhid * 8),
            #decode_layer(nhid * 8, nhid * 10, m), #nn.BatchNorm1d(nhid * 8),
            #decode_layer(nhid * 10, nhid * 12, m), #nn.BatchNorm1d(nhid * 8),
            decode_layer(nhid * 8, numNodes * 3, None)])


    def forward(self, z):
        sig = nn.Sigmoid()
        # Outputs for MSE
        XYZPred = z.clone()
        for conv in self.decoder_XYZ:
            XYZPred = conv(XYZPred)  # Graph convolution
        XYZPred = XYZPred.view(-1, self.numNodes, 3)  # Output
        
        return XYZPred.squeeze()

class CondPrior(nn.Module):
    def __init__(self, nhid, numNodes, latent_space):
        super().__init__()

        ### Conditioning network on prior for atom list
        ### Creates additional node features per node
        ### Assumes 1xself.atomRangex1 one hot encoding vector as input
        ### Output: 1x2*latent_dimx1
        self.condPrior_PDF = nn.Sequential(
            GatedConv1d(301,48,kernel_size=1,stride=1), nn.ReLU(), #nn.BatchNorm1d(48), #nn.Dropout(0.1), #nn.BatchNorm1d(24),
            GatedConv1d(48,24,kernel_size=1,stride=1), nn.ReLU(), #nn.BatchNorm1d(24),
            GatedConv1d(24,2*latent_space,kernel_size=1,stride=1))

    def forward(self, y_PDF):
        for conv in self.condPrior_PDF:
            y_PDF = conv(y_PDF)

        return y_PDF

class nodeVAE(nn.Module):
    def __init__(self, nhid, numNodesF, satellites, numNodes, atomRange, latent_space, cond_dim):
        super(nodeVAE, self).__init__()
        self.cond_dim = cond_dim
        self.numNodesF = numNodesF
        self.satellites = satellites
        self.numNodes = numNodes
        self.hidden_dim = nhid
        self.latent_space = latent_space
        self.atomRange = atomRange
       
        self.conditioning = Conditioning(self.numNodes, self.cond_dim)
        self.encoder = Encoder(self.numNodesF, self.numNodes, self.hidden_dim, self.latent_space, self.cond_dim)
        self.decoder = Decoder(self.hidden_dim, self.numNodes, self.latent_space, self.satellites)
        self.condprior = CondPrior(self.hidden_dim, self.numNodes, self.latent_space)

    def forward(self, x, y_PDF, sampling=False):
        # obtain mean and std for the conditional prior distribution
        # these parameters mu_prior, sigma_prior are conditioned on atom list/pdf
        emb_prior = self.condprior(y_PDF).transpose(1,2)
        mu_prior, log_var_prior = torch.chunk(emb_prior, 2, dim=-1)
        log_var_prior = softplus(log_var_prior)
        sigma_prior = torch.exp(log_var_prior / 2)

        if not sampling:
            # Conditioning
            condIp = self.conditioning(y_PDF)
            
            # Encoder
            mu, log_var = self.encoder(x, condIp)

            # Make the posterior model
            # Instantiate a diagonal Gaussian with mean=mu, std=sigma
            # This is the conditional posterior latent distribution q(z|x,y)
            sigma = torch.exp(log_var / 2)
            mu, sigma = mu.squeeze(), sigma.squeeze()
            kl = 0
        
        for i in range(len(mu_prior)):
            # Instantiate a diagonal Gaussian with mean=mu_0, std=sigma_0
            # This is the conditional prior distribution p(z|y)
            prior = Independent(Normal(loc=mu_prior[i],scale=sigma_prior[i]),1)
            if not sampling:
                posterior = Independent(Normal(loc=mu[i],scale=sigma[i]),1)
                # Estimate the KLD between q(z|x,y)|| p(z|y)
                kl += KLD(posterior,prior).sum()/len(mu)

                # Draw z from posterior distribution
                z_ph = posterior.rsample().unsqueeze(0)
                if i == 0:
                    z = z_ph.clone()
                else:
                    z = torch.cat((z, z_ph), dim=0)
            if sampling:
                mu = "sampling"
                log_var = "sampling"
                kl = "sampling"
                z_ph = prior.rsample()
                if i == 0:
                    z = z_ph.clone()
                else:
                    z = torch.cat((z, z_ph), dim=0)         
        
        # Decoder
        satPred = self.decoder(z)

        return mu, log_var, z, satPred, kl

class decode_layer(nn.Module):
    def __init__(self, in_feats, out_feats, act):
        super(decode_layer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.act = act
        self.out_feats = out_feats

    def forward(self, inputs):
        h = self.linear(inputs)
        if self.act != None:
            h = self.act(h)
        return h
