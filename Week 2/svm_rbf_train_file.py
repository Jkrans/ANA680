# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

df = pd.read_csv("breast-cancer-wisconsin.data", header=None, names=COLUMNS)

df = df.replace("?", pd.NA).dropna()
for col in COLUMNS:
    df[col] = pd.to_numeric(df[col])

X = df.drop(columns=["ID", "Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1721, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", random_state=1721))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))

with open("model.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Saved model.pkl")
