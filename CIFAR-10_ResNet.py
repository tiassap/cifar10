

import numpy as np
import random
import matplotlib.pyplot as plt
#import os
from tqdm import tqdm

import torch
import torch.nn as nn
#import torch.nn.functional as F

# Pytorch model ('resnet.py') by Kuang Liu: https://github.com/kuangliu/pytorch-cifar
import resnet
model = resnet.ResNet18()
#model = resnet.ResNet34()
#model = resnet.ResNet50()


# Load CIFAR-10 data. Loader script ('load_cifar_10_alt.py') by Petras Saduikis: https://github.com/snatch59/load-cifar-10
# Dataset downloaded from: http://www.cs.toronto.edu/~kriz/cifar.html
exec(open('load_cifar_10_alt.py').read())

classes= dict(zip(range(10),["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
"truck"]))

# Show an image with label inside training sample
random.seed(0)
sample_image = random.randint(0, 5000)
plt.imshow(x_train[sample_image, :, :, 0:3])
print("class : ", classes[int(y_train[sample_image, :])])

# Change the structure the image data array to (batch_size, channels, height, width)
# Change label data shape to (batch_size, )
x_train = np.moveaxis(x_train, 3, 1)
y_train = np.reshape(y_train, (y_train.shape[0],))
x_test =np.moveaxis(x_test, 3, 1)
y_test = np.reshape(y_test, (y_test.shape[0],))

# Change the channel data range to [-1, 1] (initially [0, 255])
x_train = torch.from_numpy(x_train) * (2/ 255) - 1
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test) * (2/ 255) -1 
y_test = torch.from_numpy(y_test) 

# Set GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_GPU = True

if use_GPU:
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    model = model.to(device)
    compute_by = 'GPU'
else:
    compute_by = 'CPU'


# Separate validation sample from training sample.
dev_split_index = int(9 * len(x_train) / 10)
x_dev = x_train[dev_split_index:]
y_dev = y_train[dev_split_index:]
x_train = x_train[:dev_split_index]
y_train = y_train[:dev_split_index]

# Shuffle the order of image data
permutation = torch.randperm(x_train.shape[0])
x_train = x_train[permutation].view(x_train.size())
y_train = y_train[permutation].view(y_train.size())


# Define function for training model, run epoch, and compute accuracy
# Script adapted from MITx6.86x-Machine Learning with Python-From Linear Models to Deep Learning:
# https://www.edx.org/course/machine-learning-with-python-from-linear-models-to
# 'Project＿3/mnist/part2-mnist/train_utils.py' and 'nnet_cnn.py'

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=False)
epochs = 15

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    
    return torch.mean(torch.eq(predictions, y).float())


# Training Procedure
def train_model(train_data, dev_data, model, optimizer, n_epochs, criterion):
    """Train a model for N epochs given data and hyper-params."""
        
    #for epoch in range(1, 11):
    for epoch in range(1, 1+n_epochs):
        print("-------------\nEpoch {}:\n".format(epoch))


        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer, criterion)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer, criterion)
        print('Val loss:   {:.6f} | Val accuracy:   {:.6f}'.format(val_loss, val_acc))
        
                
        # Save model
        torch.save(model, 'trained_model.pt')
    
    return val_acc
    

def run_epoch(data, model, optimizer, criterion):    
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Get x and y
        x, y = batch['x'] , batch['y'] 
                
        # Get output predictions
        out = model(x)

        # Predict and store accuracy     
        predictions = torch.argmax(out, dim=1)
        
        batch_accuracies.append(compute_accuracy(predictions, y))

        # Compute loss
        loss = criterion(out, y)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = torch.mean(torch.FloatTensor(losses))
    
    
    avg_accuracy = torch.mean(torch.FloatTensor(batch_accuracies))
    return avg_loss, avg_accuracy


batch_size = 50
def batchify_data(x_data, y_data, batch_size):
    """
    Takes a set of data points and labels and groups them into batches.
    
    Output: list, content: dictionary of 
    """
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        
        batches.append({
            'x': x_data[i:i+batch_size].clone().detach().float(),
            'y': y_data[i:i+batch_size].clone().detach().long()
        })
        
    return batches

train_batches = batchify_data(x_train, y_train, batch_size)
dev_batches = batchify_data(x_dev, y_dev, batch_size)
test_batches = batchify_data(x_test, y_test, batch_size)


# Start training
print("Start training with {}".format(compute_by))

import time
start_t = time.time()
train_model(train_batches, dev_batches, model, optimizer, epochs, criterion)

# Evaluate the model on test data
loss, accuracy = run_epoch(test_batches, model.eval(), None, criterion)
print("Loss on test set: {:.4f}. Accuracy on test set: {:.4f}".format(float(loss), float(accuracy)))

end_t = time.time()
print("\n Calculating time using {}: {:.4f} s.".format(compute_by, end_t - start_t))


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)




