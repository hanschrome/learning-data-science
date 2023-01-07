import pandas as pd

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv("datasets/dataset.csv", names=["local", "visitante", "signo", "fecha_completa"])

# Convert the signo column to a categorical data type
df["signo"] = df["signo"].astype("category")

# Encode the categorical variables as numerical data
df = pd.get_dummies(df, columns=["local", "visitante"])

# Split the data into features (X) and labels (y)
X = df.drop("signo", axis=1)
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
