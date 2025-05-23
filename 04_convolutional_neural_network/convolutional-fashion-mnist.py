
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
import time
import numpy as np

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
train_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Download and load the test data
test_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Below function can be used to visualize some of the different images from fashion mnist data

def view_fashionmnist(label, count = 1):
    fig = plt.figure()
    idx = 1
    for inputs, labels in test_loader:
        for i, input in enumerate(inputs):
            # we only want to view a certain class
            if (labels[i] != label):
                continue
            # plot the ground truth
            ax = fig.add_subplot(1, count, idx)
            input = input.cpu().detach().numpy().reshape((28,28))
            ax.imshow(input, cmap='gray')
            idx += 1
            if idx > count:
                break
        if idx > count:
            break


# View 6 bags ( label 8 is mapped to class of bags)
# View 6 types of bags
view_fashionmnist(8,6)

# Building the network
# As with MNIST, each image is 28x28 which is a total of 784 pixels, and there are 10 classes.
# Using ReLU activations for the layers

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # two convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        # Create three fully connected layers
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, X):
        # First pass
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)

        # Second pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # Re-view to flatten it out
        X = X.view(-1, 16*5*5)

        # Fully connected layers without dropout
        #X = F.relu(self.fc1(X))
        #X = F.relu(self.fc2(X))
   
        # Fully connected layers with dropout
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.dropout(F.relu(self.fc2(X)))

        X = self.fc3(X)  #no relu since it's the last one

        return F.log_softmax(X, dim=1)


torch.manual_seed(41)  # starting point, we can adjust it to test. Arbitrary value
model = ConvolutionalNetwork()

# Training the network
# Define the optimizer and loss function. 
#Choosing a smaller learning rate will make it the training run longer

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Need to track metrics for our training and testing and for time keeping
start_time = time.time()
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_correct = 0
    tst_correct = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1 # Start batches at 1
        # Get predicted values from trainign set and not flattened as it's in 2D
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)  # how off we're and compare the predictions to correct ans in y_train
        predicted = torch.max(y_pred.data, 1)[1]   # adding correct predictions, index off the first point
        batch_correct = (predicted == y_train).sum()  # how many correct we got from this batch, True=1, False=0 and sum these up
        trn_correct += batch_correct  # keep track we go along training

        # Update the parameters
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights as we go

        # Print the results every 1000 batches
        if b%1000 == 0:
            print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

        train_losses.append(loss)
        train_correct.append(trn_correct)

    # Start the tests
    with torch.no_grad(): # No gradient so we don't update our weights and biases with test data
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]  # adding correct predictions
            tst_correct += (predicted == y_test).sum() # True=1, False=0 and sum these up

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_correct)

current_time = time.time()
elapsed_time = current_time - start_time
print(f'Duration of training: {elapsed_time/60} mins ...')

# Graph the losses at each epoch. 
# Need to convert tensors to np array, otherwise we get below error
# RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

train_losses = [tloss.item() for tloss in train_losses]
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test/Validation Loss")
plt.title("Losses at each epoch")
plt.legend()
plt.show()

# Graph the accuracy at the end of each epoch
plt.plot([t/400 for t in train_correct], label="Training Accuracy")
plt.plot([t/100 for t in test_correct], label="Validation Accuracy")
plt.title("Accuracy at the end of each epoch")
plt.legend()
plt.show()


