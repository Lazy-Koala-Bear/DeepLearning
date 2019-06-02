# EE559 - Deep Learning Mini-project 1
# Written by Tianlun Luo, Yue Xu, Earvin Tio

import proj1_models as proj1
import torch
from torch.autograd import Variable
import dlc_practical_prologue as prologue
import time

# Getting our training and testing data sets
print("Getting train and test datasets")
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
mean, std = train_input.mean(), train_input.std()
test_mean, test_std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(test_mean).div_(test_std)
print("\n")

################################ Baseline model ######################################################
# set up variables for baseline model
b_train_input, b_train_target = Variable(train_input.view(-1,392)), Variable(train_target)
b_test_input, b_test_target = Variable(test_input.view(-1,392)), Variable(test_target)

baseline = proj1.create_baseline_model()

print("Training baseline model...")
start_time = time.time()
proj1.train_model(baseline, b_train_input, b_train_target)
time_elasped = time.time() - start_time

print('Baseline Model: train_error {:.02f}% test_error {:.02f}%'.format(
    proj1.compute_nb_errors(baseline, b_train_input, b_train_target) / b_train_input.size(0) * 100,
    proj1.compute_nb_errors(baseline, b_test_input, b_test_target) / b_test_input.size(0) * 100
))

print("Training time:", time_elasped)
print("\n")

################################ 100 class + logic comparison #########################################
a_train_input = Variable(train_input.view(-1,392))
a_train_labels = train_classes[:,0]*10 + train_classes[:,1]
a_train_target = Variable(a_train_labels)

a_test_input = Variable(test_input.view(-1,392))
a_test_labels = test_classes[:,0]*10 + test_classes[:,1]
a_test_target = Variable(a_test_labels)

auxiliary = proj1.create_aux_model()

print("Training 100 class + logic comparison model...")
start_time = time.time()
proj1.train_model(auxiliary, a_train_input, a_train_target)
time_elasped = time.time() - start_time

# print('100 Class + Logic Comparason: phase1_train_error {:.02f}%, phase_test_error {:.02f}%'.format(
#     proj1.compute_nb_errors(auxiliary, a_train_input, a_train_target) / a_train_input.size(0) * 100,
#     proj1.compute_nb_errors(auxiliary, a_test_input, a_test_target) / a_test_input.size(0) * 100
# ))
print("100 Class + Logic Comparason: train_error",
    proj1.compute_nb_errors_a(auxiliary, a_train_input, train_target)/train_input.size(0) * 100, "%"
)
print("100 Class + Logic Comparason: test_error",
    proj1.compute_nb_errors_a(auxiliary, a_test_input, test_target)/test_input.size(0) * 100, "%"
)
print("Training time:", time_elasped)
print("\n")

################################ 100 class + Auxiliary loss #########################################
a_train_input = Variable(train_input.view(-1,392))
a_train_labels = train_classes[:,0]*10 + train_classes[:,1]
a_train_target = Variable(a_train_labels)

a_test_input = Variable(test_input.view(-1,392))
a_test_labels = test_classes[:,0]*10 + test_classes[:,1]
a_test_target = Variable(a_test_labels)

aux_mode1_1 = proj1.create_aux_model()
aux_model_2 = proj1.create_aux_model2()

print("Training 100 class + auxiliary loss model...")
start_time = time.time()
proj1.train_model_100(aux_mode1_1, aux_model_2, a_train_input, a_train_target, train_target)
time_elasped = time.time() - start_time

print("100 Class Auxilliary Loss: train_error",
    proj1.compute_nb_errors_100(aux_mode1_1, aux_model_2, a_train_input, train_target)/train_input.size(0) * 100, "%"
)
print("100 Class Auxilliary Loss: test_error",
    proj1.compute_nb_errors_100(aux_mode1_1, aux_model_2, a_test_input, test_target)/test_input.size(0) * 100, "%"
)
print("Training time:", time_elasped)
print("\n")

################################ Weight sharing with logic comparison ################################
train_input_1 = Variable(train_input[:,0,:,:].view(-1,196))
train_input_2 = Variable(train_input[:,1,:,:].view(-1,196))

train_target_1 = Variable(train_classes[:,0])
train_target_2 = Variable(train_classes[:,1])

test_input_1 = Variable(test_input[:,0,:,:].view(-1,196))
test_input_2 = Variable(test_input[:,1,:,:].view(-1,196))

test_target_1 = Variable(test_classes[:,0])
test_target_2 = Variable(test_classes[:,1])

weight_sharing = proj1.create_weight_sharing_model()

print("Training weight sharing with logic comparison model...")
start_time = time.time()
proj1.train_model_ws(weight_sharing,
               train_input_1, train_target_1,
               train_input_2, train_target_2)
time_elasped = time.time() - start_time


print("Weight Sharing Model: train_error",
    proj1.compute_nb_errors_ws(weight_sharing, train_input_1, train_input_2, train_target)/train_input_1.size(0) * 100, "%"
)
print("Weight Sharing Model: test_error" ,
    proj1.compute_nb_errors_ws(weight_sharing, test_input_1, test_input_2, test_target)/test_input_1.size(0) * 100, "%"
)
print("Training time:", time_elasped)
print("\n")

########################## Weight sharing with logic comparison (convolution) #########################
ws_conv = proj1.ConvNet(300)

first_image = Variable(train_input[:,0,:,:].view(1000, 1, 14, 14))
second_image = Variable(train_input[:,1,:,:].view(1000, 1, 14, 14))

test_first_image = Variable(test_input[:,0,:,:].view(1000, 1, 14, 14))
test_second_image = Variable(test_input[:,1,:,:].view(1000, 1, 14, 14))

#need to convert to one hot???
first_class = Variable(train_classes[:,0])
second_class = Variable(train_classes[:,1])

print("Traning weight sharing + logical comparison with convolution model...")
start_time = time.time()
proj1.train_model_ws(ws_conv,
               first_image, first_class,
               second_image, second_class)
time_elasped = time.time() - start_time

print("Conv Weight Sharing Model: train_error",
    proj1.compute_nb_errors_ws(ws_conv, first_image, second_image, train_target)/10, "%"
)
print("Conv Weight Sharing Model: test_error" ,
    proj1.compute_nb_errors_ws(ws_conv, test_first_image, test_second_image, test_target)/10, "%"
)
print("Training time:", time_elasped)
print("\n")

########################## Weight sharing with auxiliary loss (linear) #########################
# split each training input into two separate images
train_input_1 = Variable(train_input[:,0,:,:].view(-1,196))
train_input_2 = Variable(train_input[:,1,:,:].view(-1,196))

# split the labels for the digit classes for the two images
train_classes_1 = Variable(train_classes[:,0])
train_classes_2 = Variable(train_classes[:,1])

# split each test input into two separate images
test_input_1 = Variable(test_input[:,0,:,:].view(-1,196))
test_input_2 = Variable(test_input[:,1,:,:].view(-1,196))

phase1_model = proj1.create_phase1_model()
phase2_model = proj1.create_phase2_model()

print("Training Weight sharing + auxiliary loss linear model...")
start_time = time.time()
proj1.train_ws_aux(phase1_model, phase2_model,
               train_input_1, train_classes_1,
               train_input_2, train_classes_2,
               train_target)
time_elasped = time.time() - start_time

print('Weight Sharing with Auxiliary Loss Linear Model: train_error {:.02f}% test_error {:.02f}%'.format(
    proj1.compute_nb_errors_ws_aux(phase1_model, phase2_model, train_input_1, train_input_2, train_target) / train_input.size(0) * 100,
    proj1.compute_nb_errors_ws_aux(phase1_model, phase2_model, test_input_1, test_input_2, test_target) / test_input.size(0) * 100
))
print("Training time:", time_elasped)
print("\n")

########################## Weight sharing with auxiliary loss (Convolution) #########################
# split each training input into two separate images
train_input_1 = Variable(train_input[:,0,:,:]).view(1000, 1, 14, 14)
train_input_2 = Variable(train_input[:,1,:,:]).view(1000, 1, 14, 14)

# split the labels for the digit classes for the two images
train_classes_1 = Variable(train_classes[:,0])
train_classes_2 = Variable(train_classes[:,1])

# split each test input into two separate images
test_input_1 = Variable(test_input[:,0,:,:]).view(1000, 1, 14, 14)
test_input_2 = Variable(test_input[:,1,:,:]).view(1000, 1, 14, 14)

test_classes_1 = Variable(test_classes[:,0])
test_classes_2 = Variable(test_classes[:,1])

phase1_model = proj1.ConvNet(300)
phase2_model = proj1.create_phase2_model_cnn()

print("Training Weight sharing + auxiliary loss convolution model...")
start_time = time.time()
proj1.train_ws_aux_cnn(phase1_model, phase2_model,
                   train_input_1, train_classes_1,
                   train_input_2, train_classes_2,
                   train_target)
time_elasped = time.time() - start_time

print('Weight Sharing with Auxiliary Loss Model CNN: train_error {:.02f}% test_error {:.02f}%'.format(
    proj1.compute_nb_errors_ws_aux_cnn(phase1_model, phase2_model, train_input_1, train_input_2, train_target) / train_input.size(0) * 100,
    proj1.compute_nb_errors_ws_aux_cnn(phase1_model, phase2_model, test_input_1, test_input_2, test_target) / test_input.size(0) * 100
))
print("Training time:", time_elasped)
print("\n")