# When to use each algorithm



# Simple lineal regression

* **Linear relationship**: The relationship between the independent variables and the dependent variable should be linear.
* **Multivariate normality**: The residuals (errors) of the model should be normally distributed.
* **No or little multicollinearity**: The independent variables should not be highly correlated with each other.
* **No auto-correlation**: There should not be any correlation between the residuals.
* **Homoscedasticity**: The variance of the residuals should be constant across all the independent variables.

# Multiple lineal regression


* There should be a linear relationship between the independent variables and the dependent variable
* There should be no multicollinearity among the independent variables
* The errors should be normally distributed with a mean of 0
* The variance of the errors should be constant
* The independent variables should not be correlated with the errors

# Polynomial regression

* The relationships between the features and the target variable are non-linear.
* There is a need to capture higher-order relationships between the features and the target.
* There is sufficient training data available to estimate the model parameters accurately.
* The model complexity is not a concern.
* The features are not correlated with each other, as this can lead to multicollinearity in the model.
* The residuals of the model should be approximately normally distributed, with constant variance.
* The model should be able to make good predictions on unseen data.

# Support Vector Regression (SVR)

* The data should be continuous and have a linear relationship.
* The data should be clean and free of missing values.
* The data should have a single output variable.
* The data should be suitable for the kernel function being used (linear, polynomial, or radial basis function).
* There should be enough data to accurately model the relationship between the input and output variables.

