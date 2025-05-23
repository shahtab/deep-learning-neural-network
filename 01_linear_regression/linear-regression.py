

import numpy as np
from sklearn.linear_model import LinearRegression

#Created the x,y numpy array from the dataset in the excel spreadsheet

# independent variables (x) is multi-dimensional, as I wantto find a linear relationship 
#between independent variables (x) and a single dependent variable (y)
x = np.array([[1],[-2],[3],[4.5],[0],[-4],[-1],[4],[-1]])



# y array (representing the target variable) needs to be a 1-dimensional array because it 
#typically represents a single prediction value for each data point
y = np.array([4,3,6,8,2,-3,-2,7,2.5])


# Create a Linear Regression model
model = LinearRegression()


# Train the model on the training data
model.fit(x, y)

# for the equation, y = mx + b , the slope/coefficient (m) and y-intercept (b) is obtained from the modelvariable and  method
m = model.coef_[0]
b = model.intercept_

print(f"Equation: y = {m}x + {b}")
