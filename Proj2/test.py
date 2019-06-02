# EE559 - Deep Learning Mini-project 2
# Written by Tianlun Luo, Yue Xu, Earvin Tio

import proj2_framework as proj2
import torch
import math
from torch import empty

# Generate 1000 training and testing points
train_input, train_target = proj2.generate_disc_set(1000)
test_input, test_target = proj2.generate_disc_set(1000)

# Normalize the points
train_input = train_input.sub_(train_input.mean()).div_(train_input.std())
test_input = test_input.sub_(test_input.mean()).div_(test_input.std())

train_target = proj2.convert_to_one_hot(train_target)
test_target = proj2.convert_to_one_hot(test_target)

# define our model: fully connected linear model with 3 hidden layers and 25 units each
# Activation function: ReLU for the first three layers and tanh for last layer
eta = 1e-1/100
layers = [2, 25, 25, 25, 2]
activations = ['R', 'R', 'R', 'T']
loss_model = 'CEL'
model = proj2.NNModel(layers, activations, loss=loss_model)
optimizer = proj2.optim(model.params(), eta)

mini_batch_size = 100

print("Network architecture: ")
print("Fully connected linear model")
print("Number of hidden layers:", len(layers)-2)
print("Loss function: cross entropy loss")

print("Training network...")

nb_epochs = 50
for k in range(nb_epochs):

    acc_loss = 0
    nb_train_errors = 0
    optimizer.zero_grad()

    # Train 
    for n in range(0, train_input.size(0), mini_batch_size):
        model.forward(train_input.narrow(0, n, mini_batch_size))            
        pred = torch.argmax(model.output, dim=1)
        labels = torch.argmax(train_target.narrow(0, n, mini_batch_size), dim=1)
        nb_train_errors += torch.sum(pred != labels)
        acc_loss = acc_loss + model.loss(train_target.narrow(0, n, mini_batch_size))
        model.backward(train_input.narrow(0, n, mini_batch_size), train_target.narrow(0, n, mini_batch_size))
                
        # Update parameters         
        optimizer.step()
        optimizer.zero_grad()
          
    # Test 
    nb_test_errors = 0
    for n in range(0, train_input.size(0), mini_batch_size):
        model.forward(test_input.narrow(0, n, mini_batch_size))            
        pred = torch.argmax(model.output, dim=1)
        labels = torch.argmax(test_target.narrow(0, n, mini_batch_size), dim=1)
        nb_test_errors += torch.sum(pred != labels)
        
    print('Epoch {:2d}: Train Loss {:6.2f}, Train Error: {:5.02f}%, Test Error: {:5.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))
