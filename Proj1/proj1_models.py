# EE559 - Deep Learning Mini-project 1
# Written by Tianlun Luo, Yue Xu, Earvin Tio

import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

mini_batch_size = 100

################### Compute Error functions #########################
# compute error function for baseline model 
def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        #print(output)
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


# compute error function for the 100 class + logic comparison model
def compute_nb_errors_a(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size)) 
        _, aux_class = torch.max(output.data, 1)
        tens = aux_class/10
        ones = aux_class%10
        
        predicted_classes = (tens <= ones).long()

        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


# compute error function for 100 class + auxiliary loss model
def compute_nb_errors_100(m1, m2, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output1 = m1(data_input.narrow(0, b, mini_batch_size))
        output2 = m2(output1)
        _, predicted_classes = torch.max(output2.data, 1)
 
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

# Compute error function for weight sharing + logic comparison model
def compute_nb_errors_ws(model, data_input_1, data_input_2, data_target):

    nb_data_errors = 0

    for b in range(0, data_input_1.size(0), mini_batch_size):
        output1 = model(data_input_1.narrow(0, b, mini_batch_size))
        output2 = model(data_input_2.narrow(0, b, mini_batch_size))
        _, predicted_classes1 = torch.max(output1.data, 1)
        _, predicted_classes2 = torch.max(output2.data, 1)
        predicted_classes = (predicted_classes1 <= predicted_classes2).long()

        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

# compute error function for weight sharing + Auxiliary loss model
def compute_nb_errors_ws_aux(model1, model2, data_input1, data_input2, data_target):

    nb_data_errors = 0
    mini_batch_size = 100

    for b in range(0, data_input1.size(0), mini_batch_size):
        output1 = model1(data_input1.narrow(0, b, mini_batch_size))
        output2 = model1(data_input2.narrow(0, b, mini_batch_size))
        next_input = torch.cat((output1, output2), 1)

        output = model2(next_input)
        
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

# compute error function for weight sharing + Auxiliary loss with convolution
def compute_nb_errors_ws_aux_cnn(model1, model2, data_input1, data_input2, data_target):

    nb_data_errors = 0
    mini_batch_size = 100

    for b in range(0, data_input1.size(0), mini_batch_size):
        output1 = model1(data_input1.narrow(0, b, mini_batch_size))
        output2 = model1(data_input2.narrow(0, b, mini_batch_size))
        next_input = torch.cat((output1, output2), 1)

        output = model2(next_input)
        
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


################### Train model functions #########################

# Train model function for baseline model and 100 class + logic comparison model
def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 100

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            #print(output)
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Train model function for 100 class + auxiliary loss model
def train_model_100(model, model2, train_input, train_target1, train_target2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.parameters()) + list(model2.parameters()), lr = 0.1)
    nb_epochs = 100

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output1 = model(train_input.narrow(0, b, mini_batch_size))
            output2 = model2(output1)
            #print(output)
            loss1 = criterion(output1, train_target1.narrow(0, b, mini_batch_size))
            loss2 = criterion(output2, train_target2.narrow(0, b, mini_batch_size))
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Train model function for weight sharing + logic comparison model
def train_model_ws(model, train_input_1, train_target_1, train_input_2, train_target_2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 100

    for e in range(nb_epochs):
        for b in range(0, train_input_1.size(0), mini_batch_size):
            output_1 = model(train_input_1.narrow(0, b, mini_batch_size))
            output_2 = model(train_input_2.narrow(0, b, mini_batch_size))
            loss_1 = criterion(output_1, train_target_1.narrow(0, b, mini_batch_size))
            loss_2 = criterion(output_2, train_target_2.narrow(0, b, mini_batch_size))
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# function to train the weight sharing + auxiliary model
def train_ws_aux(model_1, model_2,
                   train_input_1, train_classes_1,
                   train_input_2, train_classes_2,
                   train_target
                  ):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(list(model_1.parameters()) + list(model_2.parameters()), lr=1e-1)
    nb_epochs = 100

    mini_batch_size = 100
    
    for e in range(nb_epochs):
        for b in range(0, train_input_1.size(0), mini_batch_size):
            
            output_1 = model_1(train_input_1.narrow(0, b, mini_batch_size))
            output_2 = model_1(train_input_2.narrow(0, b, mini_batch_size))
            
            # concatenate the two outputs from the first phase
            next_input = torch.cat((output_1, output_2), 1)

            loss_1 = criterion(output_1, train_classes_1.narrow(0, b, mini_batch_size))
            loss_2 = criterion(output_2, train_classes_2.narrow(0, b, mini_batch_size))
                        
            aux_loss = loss_1 + loss_2
            
            output_3 = model_2(next_input)
            loss_3 = criterion(output_3, train_target.narrow(0, b, mini_batch_size))
            loss = aux_loss + loss_3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# function to train the weight sharing + auxiliary with convolution model
def train_ws_aux_cnn(model_1, model_2,
                       train_input_1, train_classes_1,
                       train_input_2, train_classes_2,
                       train_target
                      ):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(list(model_1.parameters()) + list(model_2.parameters()), lr=1e-1)
    nb_epochs = 100

    mini_batch_size = 100
    
    for e in range(nb_epochs):
        for b in range(0, train_input_1.size(0), mini_batch_size):
            
            output_1 = model_1(train_input_1.narrow(0, b, mini_batch_size))
            output_2 = model_1(train_input_2.narrow(0, b, mini_batch_size))
            
            next_input = torch.cat((output_1, output_2), 1)
            loss_1 = criterion(output_1, train_classes_1.narrow(0, b, mini_batch_size))
            loss_2 = criterion(output_2, train_classes_2.narrow(0, b, mini_batch_size))
                        
            aux_loss = loss_1 + loss_2
            
            output_3 = model_2(next_input)
            loss_3 = criterion(output_3, train_target.narrow(0, b, mini_batch_size))
            loss = aux_loss + loss_3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


################### Create model functions #########################

# Baseline: Fully connected linear model with 3 hidden layers, outputs 2 classes
def create_baseline_model():
    
    D_in = 2*14*14
    H1 = 100
    H2 = 50
    H3 = 20
    D_out = 2
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2, H3),
        nn.ReLU(),
        nn.Linear(H3, D_out)
    )

# 100 class + logic comparaison: fully connected linear model with 3 hidden layers,
# outputs 100 classes, final output is logically computed from the 100 classes
def create_aux_model():
    
    D_in = 2*14*14
    H1 = 100
    H2 = 50
    H3 = 20
    D_out = 100
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2, H3),
        nn.ReLU(),
        nn.Linear(H3, D_out)
    )

# 100 class + auxiliary loss: 2 linear models
# model 1: 3 hidden layers, outputs 100 classes
# model 2: 2 hidden layers, outputs 2 classes
def create_aux_model2():
    
    D_in = 100
    H1 = 50
    H2 = 20
    D_out = 2
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2, D_out)
    )


# Weight sharing with logic comparison: fully connected linear model
# with 3 hidden layers, outputs 10 classes
def create_weight_sharing_model():
    
    D_in = 14*14
    H1 = 100
    H2 = 50
    D_out = 10
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2,D_out)
    )

# CNN model: 2 convolution layers
# and 1 linear layer
class ConvNet(nn.Module):
    def __init__(self, hidden):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=1, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=1, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

# Weight sharing with auxiliary loss: 2 linear models
# model 1: 2 hidden layers, outputs 10 classes
# model 2: 1 hidden layer, outputs 2 classes

# function to create model 1
def create_phase1_model():
    
    D_in = 14*14
    H1 = 100
    H2 = 50
    D_out = 10
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2,D_out)
    )

# function to create model 2
def create_phase2_model():
    
    D_in = 20
    H1 = 200
    D_out = 2
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, D_out)

    )


# Weight sharing + auxiliary loss with convolution
# phase 1: ConvNet with 2 convolution layers and 1 linear layer
# phase 2: fully connected linear model with 2 hidden layers
# function to create the phase 2 model, phase 1 model is just the
#   ConvNet model from above
def create_phase2_model_cnn():
    
    D_in = 20
    H1 = 50
    H2 = 100
    H3 = 20
    D_out = 2
    
    return nn.Sequential(
        nn.Linear(D_in, H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2, D_out)

    )
