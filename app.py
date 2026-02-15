from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "ClumpThickness",
    "UniformityCellSize",
    "UniformityCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[f]) for f in FEATURES]
    X = np.array(values).reshape(1, -1)

    pred = model.predict(X)[0]   # should be 2 or 4
    label = "Benign (2)" if pred == 2 else "Malignant (4)"

    return render_template("index.html", result=label)

if __name__ == "__main__":
    app.run(debug=True)
