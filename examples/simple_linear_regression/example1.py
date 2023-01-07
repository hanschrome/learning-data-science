"""
Simple Linear Regression example 1

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Linear regression fit
regression = LinearRegression()
regression.fit(x_train, y_train, None)

# Predict values
y_pred = regression.predict(x_test)

# visualization
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')

plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Relation between salary and years of experience')

plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regression.predict(x_test), color='blue')

plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Relation for test data')

plt.show()

# Import the necessary metrics
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score

# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate the R2 Score
r2 = r2_score(y_test, y_pred)

# Calculate the F-Statistic
f, p = f_regression(x_test, y_test)

# Print the results
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"F-Statistic: {f}")
print(f"P-Value: {p}")
