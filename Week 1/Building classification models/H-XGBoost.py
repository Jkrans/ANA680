# H-XGBoost.py
# ANA 680 - Week 1
# Breast Cancer Wisconsin (Original) dataset
# Model: XGBoost

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

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

# XGBoost expects class labels starting at 0
# Convert 2 -> 0 (benign), 4 -> 1 (malignant)
y = y.map({2: 0, 4: 1})

# Split into train/test (25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1721, stratify=y
)

# Train the model
model = XGBClassifier(
    n_estimators=10,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("XGBoost Results")
print("Accuracy:", round(acc, 4))
print("Confusion Matrix (labels [0=Benign, 1=Malignant]):")
print(cm)
