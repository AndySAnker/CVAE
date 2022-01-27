import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb, time
from torch.autograd import Variable
torch.manual_seed(12)
SIGMA = 1
EPSILON = 1e-5

class GatedConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 kernel_size, stride, padding=0, dilation=1, activation=None):
        super(GatedConv1d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        self.h = nn.Conv1d(input_channels, output_channels, kernel_size, 
                           stride, padding, dilation)
        self.g = nn.Conv1d(input_channels, output_channels, kernel_size, 
                           stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))

        return h * g


class GatedConvTranspose1d(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 kernel_size, stride, padding=0, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose1d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose1d(input_channels, output_channels, 
                                    kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose1d(input_channels, output_channels, 
                                    kernel_size, stride, padding, output_padding,
                                    dilation=dilation)

    def forward(self, x):
        #start_time = time.time()
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        #print ("here:", time.time()-start_time)

        g = self.sigmoid(self.g(x))
        #print ("here1:", time.time()-start_time)

        return h * g

class ConvTranspose1d(nn.Module):
    def __init__(self, input_channels, output_channels, 
                 kernel_size, stride, padding=0, output_padding=0, dilation=1,
                 activation=None):
        super(ConvTranspose1d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose1d(input_channels, output_channels, 
                                    kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        
    def forward(self, x):
        #start_time = time.time()
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        #print ("here:", time.time()-start_time)

        #print ("here1:", time.time()-start_time)

        return h

class edgeGNN(nn.Module):
    def __init__(self, nfeat, nhid, nOut,nNodes, dropout, nEdgF=1):
        super(edgeGNN, self).__init__()

        self.fc_node_1_1 = nn.Linear(nfeat, nhid)
        self.fc_node_1_2 = nn.Linear(nhid, nhid)

        self.fc_edge_1_1 = nn.Linear(nhid * 2+nEdgF, nhid)
        self.fc_edge_1_2 = nn.Linear(nhid, nhid)

        self.fc_node_2_1 = nn.Linear(nhid * 2, nhid)
        self.fc_node_2_2 = nn.Linear(nhid, nhid)

        self.fc_edge_2_1 = nn.Linear(nhid * 2+nEdgF, nhid)
        self.fc_edge_2_2 = nn.Linear(nhid, nhid)


        self.ln1 = LayerNorm(nhid)
        self.ln2 = LayerNorm(nhid)
        self.dropout = dropout

        self.act = nn.ReLU()
        self.n2e = nn.Linear(2*nhid,nOut)
        self.g2e = nn.Sequential(nn.Conv1d(nNodes,int(nNodes/2),1),nn.ReLU(),
                                 nn.Conv1d(int(nNodes/2),1,1))
        self.sparseMM = SparseMM.apply

    def forward(self, x, n2e_in, n2e_out, xE):
        #pdb.set_trace()
        ## First GNN layer
        # Node MLP
        x = self.act(self.fc_node_1_1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.fc_node_1_2(x))

        # Node to edge
        x_in = self.sparseMM(n2e_in, x)
        x_out = self.sparseMM(n2e_out, x)
        x = torch.cat([x_in, x_out, xE], 1)

        # Edge MLP
        x = self.act(self.fc_edge_1_1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.fc_edge_1_2(x))
        #x = self.fc_edge_1_2(x)

        # Edge to node
        x_in = self.sparseMM(n2e_in.transpose(0, 1), x)
        x_out = self.sparseMM(n2e_out.transpose(0, 1), x)
        x = torch.cat([x_in, x_out], 1)

        
        ### Second GNN layer
        # Node MLP
        x = self.act(self.fc_node_2_1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.fc_node_2_2(x))

        # Node to edge
        x_in = self.sparseMM(n2e_in, x)
        x_out = self.sparseMM(n2e_out, x)
        x = torch.cat([x_in, x_out, xE], 1)

        # Edge MLP
        x = self.act(self.fc_edge_2_1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.fc_edge_2_2(x))
        #x = self.fc_edge_2_2(x)

        # Edge to node
        x_in = self.sparseMM(n2e_in.transpose(0, 1), x)
        x_out = self.sparseMM(n2e_out.transpose(0, 1), x)
        x = torch.cat([x_in, x_out], 1)
        

        x = self.n2e(x.unsqueeze(0))
        z = self.g2e(x)
        

        return z.squeeze(1)



class recEdgeGNN(nn.Module):
    def __init__(self, nfeat, nhid, nOut, dropout, niter):
        super(recEdgeGNN, self).__init__()

        self.fc_node_1_1 = nn.Linear(nfeat, 2*nhid)
        self.fc_node_1_2 = nn.Linear(nhid, nhid)

        self.fc_edge_1_1 = nn.Linear(nhid * 2, nhid)
        self.fc_edge_1_2 = nn.Linear(nhid, nhid)

        self.fc_node_2_1 = nn.Linear(nhid * 2, nhid)
        self.fc_node_2_2 = nn.Linear(nhid, nhid)

        self.ln1 = LayerNorm(nhid)
        self.ln2 = LayerNorm(nhid)
        self.dropout = dropout
        self.niter = niter

        self.e2p = nn.Linear(2*nhid,nOut)   # embedding to prediction

    def forward(self, x, n2e_in, n2e_out):

        x = F.relu(self.fc_node_1_1(x))

        for _ in range(self.niter):
                # Node MLP
                x = F.relu(self.fc_node_2_1(x))
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(self.fc_node_2_2(x))

                # Node to edge
                x_in = SparseMM()(n2e_in, x)
                x_out = SparseMM()(n2e_out, x)
                x = torch.cat([x_in, x_out], 1)

                # Edge MLP
                x = F.relu(self.fc_edge_1_1(x))
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(self.fc_edge_1_2(x))

                # Edge to node
                x_in = SparseMM()(n2e_in.transpose(0, 1), x)
                x_out = SparseMM()(n2e_out.transpose(0, 1), x)
                x = torch.cat([x_in, x_out], 1)
#        pdb.set_trace()
        return x, self.e2p(x.mean(0).view(1,-1))


class GraphAttentionLayer(nn.Module):
    """
    Simple Graph Attention Layer, with separate processing of self-connection.
    Equation format from https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/9_gat.html
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neighbor = nn.Parameter(
            torch.Tensor(in_features, out_features))
        self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2*out_features,1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.alpha = 0.2
        self.leakyRelu = nn.LeakyReLU(self.alpha, inplace=True)
        self.softmax = nn.Softmax(dim=1)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_neighbor.size(1))
        self.weight_neighbor.data.uniform_(-stdv, stdv)
        self.weight_self.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, n2e_in, n2e_out):
        N = adj.shape[0]

        act_self = torch.mm(input, self.weight_self)
        # Transform node activations Eq. (1)
        h = torch.mm(input, self.weight_neighbor)

        # Compute pairwise edge features (Terms inside Eq. (2))
        h_in = torch.mm(n2e_in, h)
        h_out = torch.mm(n2e_out,h)
        hEdge = torch.cat([h_in, h_out],1) 

        # Apply leakyReLU and weights for attention coefficients Eq.(2)
        e = self.leakyRelu(torch.matmul(hEdge, self.a))

        # Apply Softmax per node Eq.(3)
        # Sparse implementation
        idx = adj.coalesce().indices()
#        val = adj.coalesce().values()
        numNgbrs = (idx[0] == 0).sum()
        attention = self.softmax(e.view(-1,numNgbrs)).view(-1) 
        #pdb.set_trace()

        # Weigh nodes with attention; done by weighting the adj entries
#        alpha = torch.sparse.FloatTensor(idx,val*attention,(N,N))
        adj._values = adj._values * attention
        # Compute node updates with attention Eq. (4)
        act_neighbor = SparseMM()(adj,h)
        output = act_self + act_neighbor

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLP(nn.Module):
    def __init__(self, nfeat, nNodes, nhid, nOut,dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nhid, nhid)

        self.dropout = dropout
#        self.ln1 = LayerNorm(nhid)
        self.bn1 = nn.BatchNorm1d(nNodes)        
#        self.g2e = nn.Linear(nhid,nhid) #graph to embedding
        self.e2p = nn.Linear(nhid,nOut,bias=True)   # embedding to prediction
        self.act = nn.LeakyReLU()
#        self.act = nn.ReLU()

    def forward(self, inputs, adj):

        x = self.act(self.fc1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.fc2(x))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = self.act(self.fc3(x))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = self.act(self.fc4(x))

        return x, self.e2p(x.mean(1))

class GAT(nn.Module):
    def __init__(self, nfeat, nNodes, nhid, nOut,dropout):
        super(GAT, self).__init__()

#        self.fc1 = nn.Linear(nfeat, nhid)
#        self.fc2 = nn.Linear(nhid, nhid)

        self.gc1 = GraphAttentionLayer(nfeat, nhid)
        self.gc2 = GraphAttentionLayer(nhid, nhid)

        self.dropout = dropout
#        self.ln1 = LayerNorm(nhid)
        
#        self.g2e = nn.Linear(nhid,1) #graph to embedding
        self.e2p = nn.Linear(nhid,nOut)   # embedding to prediction


    def encode(self, x, adj, n2e_in, n2e_out):
#        pdb.set_trace()

        x = F.relu(self.gc1(x, adj, n2e_in, n2e_out))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj, n2e_in, n2e_out))

        return x

    def forward(self, inputs, adj, n2e_in, n2e_out):
        z = self.encode(inputs, adj, n2e_in, n2e_out)
#        z = self.g2e(z)
#        return z, self.e2p(z.transpose(0,1))
        return z, self.e2p(z.mean(0).view(1,-1))




class nodeGNN(nn.Module):
    def __init__(self, nfeat, nNodes, nhid, nOut,dropout):
        super(nodeGNN, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
#        self.fc2 = nn.Linear(nhid, nhid)

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
#        self.gc3 = GraphConvolutionFirstOrder(nhid, nhid)
#        self.gc4 = GraphConvolutionFirstOrder(nhid, nhid)

        self.dropout = dropout
#        self.ln1 = LayerNorm(nhid)
        
#        self.g2e = nn.Linear(nhid,1) #graph to embedding
        self.e2p = nn.Linear(nhid,nOut)   # embedding to prediction


    def encode(self, x, adj):
#        pdb.set_trace()
        x = F.relu(self.fc1(x))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = F.relu(self.fc2(x))
#        x = self.ln1(x)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = F.relu(self.gc3(x, adj))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = F.relu(self.gc4(x, adj))

        return x

    def forward(self, inputs, adj):
#        pdb.set_trace()
        z = self.encode(inputs, adj)
#        z = self.g2e(z)
#        return z, self.e2p(z.transpose(0,1))
        return z, self.e2p(z.mean(1))


class gatedGNN(nn.Module):
    def __init__(self, nfeat, nNodes, nhid, nOut,dropout):
        super(gatedGNN, self).__init__()

        self.gate1 = nn.Linear(nhid, 1)
        self.gate2 = nn.Linear(nhid, 1)

        self.gc1 = GraphConvolutionFirstOrder(nfeat, nhid)
        self.gc2 = GraphConvolutionFirstOrder(nhid, nhid)
        self.gc3 = GraphConvolutionFirstOrder(nhid, nhid)
#        self.gc4 = GraphConvolutionFirstOrder(nhid, nhid)

        self.dropout = dropout
#        self.ln1 = LayerNorm(nhid)
        
#        self.g2e = nn.Linear(nhid,1) #graph to embedding
        self.e2p = nn.Linear(nhid,nOut)   # embedding to prediction


    def encode(self, x, adj):
#        pdb.set_trace()
#        x = F.relu(self.fc1(x))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = F.relu(self.fc2(x))
#        x = self.ln1(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x_in = x
        r = F.softmax(self.gate1(x), dim=0)
        z = F.softmax(self.gate2(x), dim=0)
        x = F.relu(self.gc2(x*r, adj))
#        x = F.relu(r * x_in)
        x = (1-z) * x_in + z * x
#        x = F.relu(self.gc3(x, adj))


        return x

    def forward(self, inputs, adj):
        z = self.encode(inputs, adj)
#        pdb.set_trace()
#        z = self.g2e(z)
#        return z, self.e2p(z.transpose(0,1))
        return z, self.e2p(z.mean(0).view(1,-1))




class recGNN(nn.Module):
    def __init__(self, nfeat, nNodes, nhid, nOut,dropout,nIter,idxRange):
        super(recGNN, self).__init__()

        self.iter = nIter
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)

#        self.gc = GraphConvolutionFirstOrder(nhid, nhid)
        self.gc = GraphConvolution(nhid, nhid)

        self.dropout = dropout
#        self.e2p = nn.Linear(nhid,nOut)   # embedding to prediction
        self.e2p = nn.Linear(nhid,nOut)   # embedding to regression output
        self.g2e = nn.Linear(8,1)
        self.idxRange = idxRange
        self.ln1 = LayerNorm(nhid)
        self.ln2 = LayerNorm(nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
#        self.act = nn.ELU()
#        self.act = nn.LeakyReLU()
        self.act = nn.ReLU()
#        self.outAct = nn.Hardtanh()

    def encode(self, x, adj):
#        pdb.set_trace()
        x = self.act(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
#        x = self.ln1(x)
        x = self.act(self.fc2(x))

        for _ in range(self.iter):
            x = self.act(self.gc(x, adj))

        return x

    def decode(self,x):

        return (self.e2p(x.mean(1)))

    def multiGraphDecode(self,x):
        lIdx = self.idxRange[0,0]
        uIdx = self.idxRange[1,0]
        z = x[lIdx:uIdx].mean(1).view(1,-1)

#        pdb.set_trace()
        for i in range(1,self.idxRange.shape[1]):
            lIdx = self.idxRange[0,i]
            uIdx = self.idxRange[1,i]
            z = torch.cat((z, x[lIdx:uIdx].mean(0).view(1,-1)))

        z = self.ln2(z)
        z = F.relu(self.g2e(z.transpose(0,1)).transpose(0,1))

        return self.e2p(z).view(1,-1)

    def forward(self, inputs, adj):
#        pdb.set_trace()
        emb = self.encode(inputs, adj)
        z = self.decode(emb)
#        z = self.multiGraphDecode(emb)

        return emb, z 


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, matrix1, matrix2):
        ctx.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)
    
    @staticmethod
    def backward(ctx, grad_output):
        matrix1, matrix2 = ctx.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if ctx.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class SparseMMnonStat(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
#        if input.dim() == 3:
        support = torch.mm(input.view(-1,input.shape[-1]),self.weight)
        #output = SparseMM()(adj, support)
        sparseMM = SparseMM.apply
        output = sparseMM(adj,support)
        output = output.view(input.shape)
#    else:
#            support = torch.mm(input, self.weight)
#            output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphConvolutionFirstOrder(nn.Module):
    """
    Simple GCN layer, with separate processing of self-connection
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionFirstOrder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neighbor = nn.Parameter(
            torch.Tensor(in_features, out_features))
        self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_neighbor.size(1))
        self.weight_neighbor.data.uniform_(-stdv, stdv)
        self.weight_self.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        act_self = torch.mm(input, self.weight_self)
        support_neighbor = torch.mm(input, self.weight_neighbor)
        act_neighbor = SparseMM()(adj, support_neighbor)
        output = act_self + act_neighbor
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta