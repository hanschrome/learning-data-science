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
