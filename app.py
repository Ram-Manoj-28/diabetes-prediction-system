
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("artifacts/model.joblib")
imputer = joblib.load("artifacts/imputer.joblib")

FEATURES = [
    "pregnancies", "glucose", "bloodpressure", "skinthickness",
    "insulin", "bmi", "diabetespedigreefunction", "age"
]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[name]) for name in FEATURES]

            if any(value < 0 for value in values):
                raise ValueError()

            data = pd.DataFrame([values], columns=FEATURES)
            transformed = imputer.transform(data)

            prediction = int(model.predict(transformed)[0])
            probability = round(
                float(model.predict_proba(transformed)[0][1]) * 100, 2
            )

            result = (
                "Diabetes Positive"
                if prediction == 1
                else "Diabetes Negative"
            )

        except (ValueError, TypeError, KeyError):
            error = "Please enter valid non-negative numerical values."

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
