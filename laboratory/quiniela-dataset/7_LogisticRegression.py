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
from sklearn.linear_model import LogisticRegression

# create an instance of the LogisticRegression class
model = LogisticRegression(random_state=42)

# fit the model to the training data
model.fit(X_train, y_train)

# predict on the test set
y_pred = model.predict(X_test)

# predict probabilities
y_probs = model.predict_proba(X_test)

print(y_probs)

# Compute the MSE and R-squared of the model's predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MSE: {mse:.2f}")  # 1.29
print(f"R-squared: {r2:.2f}")  # -0.94

"""

The MSE is 1.29 which is not a meaningful metric to evaluate the performance of the model in classification problem.
The R-squared value is -0.94, which is less than 0, which means that the model's predictions are worse than just using
 the mean of the target variable. However, as in the case of MSE, R-squared is not a meaningful metric to evaluate the
  performance of the model in classification problem.
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Compute accuracy
acc = accuracy_score(y_test, y_pred)

# Compute precision
prec = precision_score(y_test, y_pred, average="macro")

# Compute recall
rec = recall_score(y_test, y_pred, average="macro")

# Compute f1-score
f1 = f1_score(y_test, y_pred, average="macro")

# Print the results
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-score: {f1:.2f}")

"""
The results you're seeing are the accuracy, precision, recall and f1-score, which are commonly used metrics to evaluate
 the performance of a classification model.

The accuracy of the model is 0.46, which means that 46% of the predictions are correct, and 54% are incorrect.
 While accuracy might be a good metric in some cases, in other cases, it is not enough by itself.
The precision is 0.21, which means that 21% of the positive predictions are correct.
 Precision is a measure of the ability of the model to avoid false positives.
  A low precision means that the model has a high rate of false positives.
The recall is 0.33, which means that the model correctly identified 33% of the actual positive samples.ç
 Recall is a measure of the ability of the model to find all the positive samples.
  A low recall means that the model has a high rate of false negatives.
The F1-score is 0.21, which is the harmonic mean of precision and recall.
 The F1-score is a balance between precision and recall,
  and it is commonly used as a metric to evaluate a model's performance.
"""
