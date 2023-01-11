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
from sklearn.ensemble import RandomForestRegressor

# Create an instance of the RandomForestRegressor class
model = RandomForestRegressor(random_state=42, n_estimators=100)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the MSE and R-squared of the model's predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MSE: {mse:.2f}")  # 0.74
print(f"R-squared: {r2:.2f}")  # -0.12

"""
The results you're getting with the random forest regressor are the MSE (Mean Squared Error) and the R-squared values of your model's predictions.

The MSE is 0.74, which means that on average, your model's predictions are off by 0.74. This value is not as small as desired, but it is not so bad either. However, the value of MSE by itself is not sufficient to judge the overall performance of the model and it's performance should be compared to other models and interpret it in the context of the problem and the data.

The R-squared value is -0.12, which is less than 0, which means that the model's predictions are worse than just using the mean of the target variable. The R-squared value is a measure of how much of the variance in the target variable is explained by the model. A negative value means that the model is not fitting the data well. A value of 0 means that the model is no better than a horizontal line, and a value of 1 means that the model perfectly predicts the target variable.

This results may be considered not good as it indicates that the model is a poor fit for the data, however, it is possible that the data or the problem is difficult to be modeled or that the model or the features are not appropriate for the problem. It is important to evaluate other models, features, or even tuning the hyperparameters of the model in order to improve the performance.

It would be useful to have domain knowledge of the problem, to make an observation about this values and make a decision about the next step. Additionally, it is important to keep in mind that one model cannot always solve every problem.
"""
