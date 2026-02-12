# F-DecisionTree.py
# ANA 680 - Week 1
# Breast Cancer Wisconsin (Original) dataset
# Model: Decision Tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Column names from the UCI dataset documentation
COLUMNS = [
    "ID",
    "ClumpThickness",
    "UniformityCellSize",
    "UniformityCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses",
    "Class"
]

# Load the .data file (comma-separated, no header)
df = pd.read_csv("breast-cancer-wisconsin.data", header=None, names=COLUMNS)

# Replace '?' with NA, then drop rows with missing values
df = df.replace("?", pd.NA).dropna()

# Convert columns to numeric after cleaning
for col in COLUMNS:
    df[col] = pd.to_numeric(df[col])

# Features and target
X = df.drop(columns=["ID", "Class"])
y = df["Class"]

# Split into train/test (25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1721, stratify=y
)

# Train the model
model = DecisionTreeClassifier(random_state=1721)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[2, 4])

print("Decision Tree Results")
print("Accuracy:", round(acc, 4))
print("Confusion Matrix (labels [2, 4]):")
print(cm)
