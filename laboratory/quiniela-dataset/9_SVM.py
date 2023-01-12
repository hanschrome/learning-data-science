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
    Training the model
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.svm import SVC

# Create a SVM classifier with a linear kernel
svm = SVC(kernel='linear', C=1)

# Fit the classifier to the training data
svm.fit(X_train, y_train)

# make predictions on the testing data
y_pred = svm.predict(X_test)

# Compute the accuracy, precision, recall and f1-score of the model's predictions
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

# Print the results
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-score: {f1:.2f}")

"""
Taking too much time...
"""