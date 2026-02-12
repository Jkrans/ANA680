# Jeremy Krans
# ChurnANNClassifier.py
# ANA 680 - Week 1
# Churn_Modelling.csv
# Goal: find important features + build a 3-layer ANN to predict Exited

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# Load data
df = pd.read_csv("Churn_Modelling.csv")

# Exited is the target (0 = stayed, 1 = left)
y = df["Exited"].astype(int)

# drop id type columns and target
X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"], errors="ignore")


# Encode categorical + scale numeric
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype != "object"]

preprocessor = ColumnTransformer(

    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop",
)

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1721, stratify=y
)

# Fit/transform
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Convert to dense arrays for Keras
if hasattr(X_train_proc, "toarray"):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()



# Feature significance
lr = LogisticRegression(
    max_iter=5000,
    solver="saga",
    penalty="l2",
    C=1.0,
    n_jobs=-1,
    random_state=1721
)
lr.fit(X_train_proc, y_train)

# Get feature names after encoding
ohe = preprocessor.named_transformers_["cat"]
cat_feature_names = []
if len(cat_cols) > 0:
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

feature_names = cat_feature_names + num_cols

coefs = lr.coef_.ravel()
imp_df = pd.DataFrame(
    {"feature": feature_names, "abs_coef": np.abs(coefs), "coef": coefs}
).sort_values("abs_coef", ascending=False)

print("\nTop 15 features by |logistic regression coefficient:")
print(imp_df.head(15).to_string(index=False))



# Build a 3-layer ANN (2 hidden layers + output layer)
tf.random.set_seed(1721)
np.random.seed(1721)

# Keras runs better with float32
X_train_ann = X_train_proc.astype("float32")
X_test_ann = X_test_proc.astype("float32")

model = Sequential()
model.add(Input(shape=(X_train_ann.shape[1],)))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the ANN
model.fit(X_train_ann, y_train, epochs=30, batch_size=32, verbose=0)

# Predict
y_prob = model.predict(X_test_ann, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# Confusion matrix + accuracy
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
acc = accuracy_score(y_test, y_pred)

print("\nANN Results")
print("Accuracy:", round(acc, 4))
print("Confusion Matrix (labels [0=Stayed, 1=Exited]):")
print(cm)
