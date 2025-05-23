import numpy as np
import pandas as pd

red_df = pd.read_csv('winequality-red.csv', sep = ";")

# Make the inputs with all cols except the last which is 'quality'
# Make quality column values binary based on the values it contains
# If value is greater than 5 then make it 1 else 0

X = red_df.drop('quality', axis = 1)
y = red_df['quality'].apply(lambda row: 1 if row > 5 else 0)

# Convert dataframe to numpy array
features = np.array(X)
labels = np.array(y)

# Set learning rate and epochs
learning_rate = 0.01
epochs = 100


# Now we modify our implementation to create an abstract SingleNeuronModel
# class that will be the parent of our classification models
# This class definition has already been given in the class notes
class SingleNeuronModel():
    def __init__(self, in_features):
        # self.w = np.zeros(in_features)
        # self.w_0 = 0.
        # Better, we set initial weights to small normally distributed values.
        self.w = 0.01 * np.random.randn(in_features)
        self.w_0 = 0.01 * np.random.randn()
        self.non_zero_tolerance = 1e-8  # add this to divisions to ensure we don't divide by 0

    def forward(self, x):
        # Calculate and save the pre-activation z
        self.z = x @ self.w.T + self.w_0

        # Apply the activation function, and return
        self.a = self.activation(self.z)
        return self.a

    def activation(self, z):
        raise ImplementationError("activation method should be implemented by subclass")

    # calculate and save gradient of our output with respect to weights
    def gradient(self, x):
        raise ImplementationError("gradient method should be implemented by subclass")

    # update weights based on gradients and learning rate
    def update(self, grad_loss, learning_rate):
        model.w -= grad_loss * self.grad_w * learning_rate
        model.w_0 -= grad_loss * self.grad_w_0 * learning_rate

class SingleNeuronClassificationModel(SingleNeuronModel):
    # Sigmoid activation function for classification
    def activation(self, z):
        return 1 / (1 + np.exp(-z) + self.non_zero_tolerance)

    # Gradient of output w.r.t. weights, for sigmoid activation
    def gradient(self, x):
        self.grad_w = self.a * (1 - self.a) * x
        self.grad_w_0 = self.a * (1 - self.a)



# Test: classification model output for a single 2D datapoint:
model = SingleNeuronClassificationModel(in_features=len(features[0]))
model.w = np.array(X)
model.w_0 = -8

x = np.array(X)
y = model.forward(x)
print("input_data", x, "=> output", y)



