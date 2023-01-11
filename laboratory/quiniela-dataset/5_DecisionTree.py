import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("datasets/output.csv", names=["local", "visitante", "signo", "fecha_completa"])

# Convert the signo column to a categorical data type
df["signo"] = df["signo"].astype("category")

# Use LabelEncoder to convert the signo column to numerical values
label_encoder = LabelEncoder()
df["signo"] = label_encoder.fit_transform(df["signo"])

# Encode the categorical variables as numerical data
df = pd.get_dummies(df, columns=["local", "visitante"])

# Split the data into features (X) and labels (y)
X = df.drop(["signo"], axis=1)
y = df["signo"]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Print the value counts of the signo column in the training and testing sets
print("\nSigno value counts in y_train:")
print(y_train.value_counts())
print("\nSigno value counts in y_test:")
print(y_test.value_counts())

print(X_train)

"""
    Decision Tree
"""

from sklearn.tree import DecisionTreeRegressor

# Create an instance of the DecisionTreeRegressor class
model = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the MSE and R-squared of the model's predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MSE: {mse:.2f}")  # 1.28
print(f"R-squared: {r2:.2f}")  # -0.93

"""

Results explained by chat-gpt

The results you're getting with the decision tree regressor are the MSE (Mean Squared Error) and the R-squared values of your model's predictions.

The MSE is 1.28, which means that on average, your model's predictions are off by 1.28. This is a relatively small value, which suggests that the model's predictions are relatively accurate. However, the value of MSE by itself is not sufficient to judge the overall performance of the model and it's performance should be compared to other models and interpret it in the context of the problem and the data.

The R-squared value is -0.93, which is less than 0, which means that the model's predictions are worse than just using the mean of the target variable. The R-squared is a measure of how much of the variance in the target variable is explained by the model. A negative value means that the model is not fitting the data well. A value of 0 means that the model is no better than a horizontal line, and a value of 1 means that the model perfectly predicts the target variable.

One possible reason for the negative R-squared value and poor predictions is that the data may be too random, meaning that it is difficult to identify any patterns in the data that can be used to make predictions. The decision tree may also be overfitting the data because of how complex is the tree, this can be solved by tuning the max_depth, min_samples_split, max_leaf_nodes, or by using pruning techniques.

Also, it could be beneficial to try other models, features, or even tuning the hyperparameters of the model to improve the performance. It would be useful to have domain knowledge of the problem, to make an observation about this values and make a decision about the next step.

"""
