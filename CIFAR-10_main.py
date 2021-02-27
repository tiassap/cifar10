import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
# import os


"""
Pytorch model ('resnet.py') by Kuang Liu: https://github.com/kuangliu/pytorch-cifar 
"""

import resnet
model = resnet.ResNet18()
#model = resnet.ResNet34()
#model = resnet.ResNet50()

"""
Load CIFAR-10 data. Loader script ('load_cifar_10_pytorch.py') adapted from 
Petras Saduikis (https://github.com/snatch59/load-cifar-10 ), adding function 
to arrange image data structure for PyTorch use.
Dataset downloaded from: http://www.cs.toronto.edu/~kriz/cifar.html
"""

from load_cifar_10_pytorch import *
cifar_10_dir = 'cifar-10-batches-py' 
(x_train, y_train), (x_test, y_test) = load_data(cifar_10_dir)
(x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train), preprocess(x_test, y_test)


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


# Function to monitor GPU memory
from subprocess import check_output
def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(check_output(COMMAND.split()))[1:]
  memory_free_values = int(memory_free_info[0].split()[0])
  print("\n Remaining GPU memory: {} MB \n".format(memory_free_values))
  #return memory_free_values


"""
Define function for training model, run epoch, and compute accuracy.
Script adapted from MITx6.86x-Machine Learning with Python-From Linear Models to Deep Learning:
https://www.edx.org/course/machine-learning-with-python-from-linear-models-to
'Project_3/mnist/part2-mnist/train_utils.py' and 'nnet_cnn.py'.
"""

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=False)
epochs = 11

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    
    return torch.mean(torch.eq(predictions, y).float())


# Training Procedure
def train_model(train_data, dev_data, model, optimizer, n_epochs, criterion):
    """Train a model for N epochs given data and hyper-params."""
        
    
    for epoch in range(1, 1+n_epochs):
        print("________________________ \n \n Epoch {}: \n".format(epoch))
        
        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer, criterion)
        print('\n Train loss: {:.6f} | Train accuracy: {:.4f}%'.format(loss, acc*100))
        
        if use_GPU:
            get_gpu_memory()

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer, criterion)
        print('\n Val loss:   {:.6f} | Val accuracy:   {:.4f}%'.format(val_loss, val_acc*100))
        
        if use_GPU:
            get_gpu_memory()
              
        # Shuffle the train-val dataset (cross validation)
        arrange_train_val_data(x_train, y_train)
                        
        # Save model
        torch.save(model, 'trained_model.pt')
        
        # Append loss and accuracy to plot
        training_accuracy.append(acc*100)
        training_loss.append(loss)
        validation_accuracy.append(val_acc*100)
        validation_loss.append(val_loss)
        run_epochs.append(epoch)
    
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
    
    Output: list; list_content: dictionary of tensor x and tensor y
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



def arrange_train_val_data(x, y):
    """
    Shuffle (randomly) and split training data into 
    training & validation data for cross-validation. 
    """
    # global x_train, y_train, x_dev, y_dev 
    global train_batches, dev_batches
    
    # Shuffle the order of image data
    permutation = torch.randperm(x.shape[0])
    x_train = x[permutation].view(x.size())
    y_train = y[permutation].view(y.size())
    
    # Separate validation sample from training sample.
    dev_split_index = int(9 * len(x) / 10)
    x_dev = x[dev_split_index:]
    y_dev = y[dev_split_index:]
    x_train = x[:dev_split_index]
    y_train = y[:dev_split_index]
    
    train_batches = batchify_data(x_train, y_train, batch_size)
    dev_batches = batchify_data(x_dev, y_dev, batch_size)
    
    # delete local variable to save GPU memory
    del x_dev, y_dev, x_train, y_train



train_batches, dev_batches= [], []
test_batches = batchify_data(x_test, y_test, batch_size)
arrange_train_val_data(x_train, y_train)
training_accuracy, training_loss, validation_accuracy, validation_loss, run_epochs = [], [], [], [], []


# Start training
print("\n \n Start training with {} \n ".format(compute_by))

from time import time
start_t = time()
train_model(train_batches, dev_batches, model, optimizer, epochs, criterion)

# Evaluate the model on test data
loss, accuracy = run_epoch(test_batches, model.eval(), None, criterion)
print("\n ==================== \n Loss on test set: {:.4f}. Accuracy on test set: {:.4f}%".format(float(loss), float(accuracy*100)))

end_t = time()
print("\n Calculation time using {}: {:.4f} s.".format(compute_by, end_t - start_t))

from datetime import datetime
FORMAT = '%Y%m%d%H%M%S'
datenow = datetime.now().strftime(FORMAT)
PATH = './cifar_net.pth_%s'%(datenow)
torch.save(model.state_dict(), PATH)


# Plot loss and accuracy
f = plt.figure(figsize=(10,3))
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)

ax1.plot(run_epochs, training_loss, label="training")
ax1.plot(run_epochs, validation_loss, label="validation")
ax1.set_title('Loss')
ax1.legend()

ax2.plot(run_epochs, training_accuracy, label="training")
ax2.plot(run_epochs, validation_accuracy, label ="validation")
ax2.set_title('Accuracy (%)')
ax2.legend()





