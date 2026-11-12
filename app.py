
import os
import json
import logging

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from flask import Flask, render_template, request

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2")
ENDPOINT_NAME = os.environ.get(
    "SAGEMAKER_ENDPOINT_NAME",
    "diabetes-prediction-endpoint"
)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)

FEATURES = [
    "pregnancies",
    "glucose",
    "bloodpressure",
    "skinthickness",
    "insulin",
    "bmi",
    "diabetespedigreefunction",
    "age"
]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            values = []

            for name in FEATURES:
                raw_value = request.form.get(name, "").strip()

                form_data[name] = raw_value

                if not raw_value:
                    raise ValueError(
                        "Please complete all eight fields."
                    )

                value = float(raw_value)

                if value < 0:
                    raise ValueError(
                        "Values cannot be negative."
                    )

                if not (value < float("inf")):
                    raise ValueError(
                        "Please enter valid numerical values."
                    )

                values.append(value)

            payload = {
                "instances": [values]
            }

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload).encode("utf-8")
            )

            prediction = json.loads(
                response["Body"].read().decode("utf-8")
            )

            predicted_class = int(
                prediction["predictions"][0]
            )

            probability = float(
                prediction["diabetes_probability"][0]
            ) * 100

            if predicted_class not in (0, 1):
                raise ValueError("Unexpected prediction returned.")

            if not 0 <= probability <= 100:
                raise ValueError("Invalid probability returned.")

            result = {
                "prediction": predicted_class,
                "probability": round(probability, 2),
                "label": (
                    "Diabetes Positive"
                    if predicted_class == 1
                    else "Diabetes Negative"
                )
            }

        except ValueError as e:
            error = str(e)

        except (ClientError, BotoCoreError):
            app.logger.exception("AWS SageMaker request failed")
            error = (
                "The prediction service is temporarily unavailable. "
                "Please try again later."
            )

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "An unexpected error occurred. "
                "Please try again later."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_data=form_data
    )


@app.route("/health")
def health():
    return {
        "status": "running",
        "endpoint": ENDPOINT_NAME,
        "region": REGION
    }, 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
