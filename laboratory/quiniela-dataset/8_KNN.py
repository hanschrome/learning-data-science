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
    KNN
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# create a k-NN classifier with k = 5
knn = KNeighborsClassifier(n_neighbors=5)

# fit the classifier to the training data
knn.fit(X_train, y_train)

# make predictions on the testing data
y_pred = knn.predict(X_test)

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

The results you've obtained with K-NN indicate that the model has an accuracy of 40%. 
This means that 40% of the predictions made by the model are correct, while 60% are incorrect.

The precision is 0.33, which means that 33% of the positive predictions made by the model are correct. 
A low precision means that the model has a high rate of false positives.

The recall is 0.33, which means that the model correctly identified 33% of the actual positive samples. 
A low recall means that the model has a high rate of false negatives.

The F1-score is 0.31, which is the harmonic mean of precision and recall. 
It is a balance between precision and recall and a commonly used metric to evaluate the performance of a model.

A low accuracy, precision, recall, and F1-score indicate that the model is not performing well. 
It could be that the k-NN algorithm is not suitable for this dataset or that the hyperparameter k is not well-tuned. 
Additionally, it could also be that there is a problem with the data.

It's important to note that these values can change depending on the value of k, so you should experiment with
 different values of k to see which one gives the best results. Also, it's a good practice to test the performance of
  the model using cross-validation to get a better estimate of the model's performance on unseen data.

"""
