# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load data from the CSV file
data = pd.read_csv('Sales.csv')

# Extracting features (X) and target variable (y) from the dataset
X = data['TV'].values.reshape(-1, 1)  # Reshape X to a 2D array
y = data['Sales'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test data
predictions = model.predict(X_test)

# Predicting for a single value (e.g., 100)
predicted_data = model.predict(np.array([[150]]))

print("Predicted value:", predicted_data)

# Plotting the original data and the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('Simple Linear Regression')
plt.show()
