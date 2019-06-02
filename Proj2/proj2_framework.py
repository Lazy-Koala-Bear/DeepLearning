# EE559 - Deep Learning Mini-project 2
# Written by Tianlun Luo, Yue Xu, Earvin Tio

import torch
import math
from torch import empty

torch.set_grad_enabled(False)

# Function to generate train and test datasets
def generate_disc_set(points):
    # squared both sides
    # x^2 + y^2 - 1/2pi
    # if negative -> inside; if positive -> outside
    # need the opposite
    # if negative -> outside; if positive -> inside
    # add 1 and div by 2 moves from [-1, 0, 1] to [0, 0.5, 1]
    # applying ceil moves from [0, 0.5, 1] to [0, 1, 1]
    # so points on the disc itself (radius of 1/2pi) will be considered inside

    _input = torch.empty(points, 2).uniform_(0, 1)
    _target = _input.pow(2).sum(1).sub(1 / (2*math.pi)).sign().neg().add(1).div(2).ceil()
    return _input, _target    


# helper function to convert target vector to one-hot encoding
def convert_to_one_hot(data):
    onehot = torch.zeros(1000,2)
    for i in range(1000):
        onehot[i][data[i].long()] = 1   
    return onehot

# Framework class
class NNModel:
    def __init__(self, layers, activations, loss="MSE"):

        self.activations  = activations
        self.loss_mode    = loss
        self.layers       = layers            
        self.list_size    = len(layers)-1

        # initialize parameters
        self.initialize()
        
    def initialize(self):
        self.w_list       = [0]*self.list_size
        self.b_list       = [0]*self.list_size
        self.d_w_list     = [0]*self.list_size
        self.d_b_list     = [0]*self.list_size
                
        for i in range(self.list_size):
            var = 1.0/((self.layers[i] + self.layers[i+1])/2)
            self.w_list[i] = torch.empty(self.layers[i+1], self.layers[i]).normal_(0, var)
            self.d_w_list[i] = torch.empty(self.layers[i+1], self.layers[i]).normal_(0, var)
            self.b_list[i] = torch.empty(self.layers[i+1]).normal_(0, var)
            self.d_b_list[i] = torch.empty(self.layers[i+1]).normal_(0, var)
            
    def params(self):
        return (self.w_list, self.b_list, self.d_w_list, self.d_b_list)
        
                    
    def forward(self, x):
        self.x_list = [0]*self.list_size
        self.s_list = [0]*self.list_size
        
        self.s_list[0] = x.mm(self.w_list[0].t()) + self.b_list[0]
        self.x_list[0] = self.act(self.s_list[0], self.activations[0])
        for i in range(self.list_size - 1):
            self.s_list[i+1] = self.x_list[i].mm(self.w_list[i+1].t()) + self.b_list[i+1]
            self.x_list[i+1] = self.act(self.s_list[i+1], self.activations[i+1])
        
        self.output = self.x_list[self.list_size - 1]

        
    def backward(self, x, t):
        
        self.dl_dx_list = [0]*self.list_size
        self.dl_ds_list = [0]*self.list_size
        
        self.dl_dx_list[self.list_size - 1] = self.d_loss(t)
        self.dl_ds_list[self.list_size - 1] = self.d_act(self.s_list[self.list_size - 1], self.activations[self.list_size-1]) * self.dl_dx_list[self.list_size - 1]

        
        for i in range(self.list_size - 1):
            curr_idx = self.list_size - 2 - i
            self.dl_dx_list[curr_idx] = self.dl_ds_list[curr_idx+1].mm(self.w_list[curr_idx+1])
            self.dl_ds_list[curr_idx] = self.d_act(self.s_list[curr_idx], self.activations[curr_idx]) * self.dl_dx_list[curr_idx]
            
        for i in range(self.list_size - 1):
            curr_idx = self.list_size - 1 - i
            self.d_w_list[curr_idx].add_(self.dl_ds_list[curr_idx].t().mm(self.x_list[curr_idx - 1]))
            self.d_b_list[curr_idx].add_(torch.sum(self.dl_ds_list[curr_idx], dim=0))
            
        self.d_w_list[0].add_(self.dl_ds_list[0].t().mm(x))
        self.d_b_list[0].add_(torch.sum(self.dl_ds_list[0], dim=0))

        
    def act(self, x, activation):
        if activation == "T":
            return self.tanh(x)
        elif activation == 'R':
            return self.relu(x)
        elif activation == 'S':
            return self.sigmoid(x)
        else:
            return x
    
    def d_act(self, x, activation):
        if activation == "T":
            return self.d_tanh(x)
        elif activation == 'R': 
            return self.d_relu(x)
        elif activation == 'S':
            return self.d_sigmoid(x)
        else:
            return 1
    
    def loss(self, t):
        if self.loss_mode == "MSE":
            return self.mse_loss(self.output, t)
        else:
            return self.cross_entropy_loss(self.output, t)
        
    def d_loss(self, t):
        if self.loss_mode == "MSE":
            return self.d_mse_loss(self.output, t)
        else:
            return self.d_cross_entropy_loss(self.output, t)
    
    
    def tanh(self, x):
        return x.tanh()
    
    def d_tanh(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
    
    def relu(self, x):
        x[x<=0] = 0
        return x
    
    def d_relu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def d_sigmoid(self, x):
        return x * (1 - x)
    
    def mse_loss(self, v, t):
        return (v - t).pow(2).sum()
        
    def d_mse_loss(self, v, t):
        return 2 * (v - t)
    
    def cross_entropy_loss(self, x, y):

        m = y.shape[0]
        p = self.softmax(x)
        ce = 0
        for i in range(m):
            ce = ce + -torch.sum(y[i]*torch.log(p[i]+1e-9))/p[i].shape[0]

        return ce/m

    def d_cross_entropy_loss(self, x, y):
        m = y.shape[0]
        g = self.softmax(x)
        for i in range(m):
            g[i][torch.argmax(y[i])] -= 1
        return g

    def softmax(self, x):
        for i in range(x.shape[0]):
            e = torch.exp(x[i] - torch.max(x[i], 0)[0].item())
            x[i] = e / torch.sum(e)
        return x

# optimizer class
class optim:
    def __init__(self, params, lr):
        self.w_list    = params[0]
        self.b_list    = params[1]
        self.d_w_list  = params[2]
        self.d_b_list  = params[3]
        self.lr = lr
        
    def zero_grad(self):
        for i in range(len(self.d_w_list)):
            self.d_w_list[i].zero_()
            self.d_b_list[i].zero_()
            
    def step(self):
        for i in range(len(self.w_list)):
            self.w_list[i] = self.w_list[i] - self.lr * self.d_w_list[i]
            self.b_list[i] = self.b_list[i] - self.lr * self.d_b_list[i]


