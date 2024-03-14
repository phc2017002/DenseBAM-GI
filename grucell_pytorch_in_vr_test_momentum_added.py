import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from locked_dropout_test import LockedDropout
import math
import torch.nn.functional as F


class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, mu, epsilon, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.mu = mu
        self.epsilon = epsilon
        self.drop_layer = LockedDropout()
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
        #self.batch_first = batch_first


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden, v):
        
        x = x.view(-1, x.size(1))

        #x = self.drop_layer(x,0.2)


        #print("x.size() in GRUCell",x.size())

        #print("batch_first",batch_first)

        gate_x = self.x2h(x)

        #print("gate_x.size()",gate_x.size())

        #print("gate_x.size()",gate_x.size())
        #print("v.size()",v.size())
        
        vy = self.mu*v + self.epsilon*gate_x
        gate_h = self.h2h(hidden) + vy
        
        #gate_h = self.h2h(hidden)
        
        #print('gate_x',gate_x)
        #print('gate_h.shape',gate_h.shape)

        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()

        #print('gate_x.shape',gate_x.shape)
        #print('gate_h.shape',gate_h.shape)

        i_r, i_i, i_n = gate_x.chunk(3,1)
        h_r, h_i, h_n = gate_h.chunk(3,1)
        
        #print("i_r.shape",i_r.shape)
        #print("h_r.shape",h_r.shape)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        #print('hidden.shape',hidden.shape)
        
        hy = newgate + inputgate * (hidden - newgate)
        
        #hy_d = self.drop_layer(hy,0.2)

        
        return hy, vy
