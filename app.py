"""Small Flask interface for the Titanic survival model."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def load_model():
    for candidate in (BASE_DIR / "titanic_model.pkl", BASE_DIR / "model.pkl"):
        if candidate.is_file():
            try:
                with candidate.open("rb") as model_file:
                    return pickle.load(model_file)
            except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError, ValueError) as exc:
                raise RuntimeError(f"Could not load model file {candidate.name}: {exc}") from exc
    raise RuntimeError("Model file not found. Add titanic_model.pkl or model.pkl beside app.py.")


model = load_model()
EMPTY_CONTEXT = {
    "values": {}, "prediction": None, "result": None,
    "survived_probability": 0, "not_survived_probability": 0,
    "confidence": None, "family_size": None, "error": None,
}


def render_home(**context):
    return render_template("index.html", **{**EMPTY_CONTEXT, **context})


@app.get("/")
def home():
    return render_home()


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None)


@app.post("/predict")
def predict():
    try:
        raw = {key: request.form.get(key, "").strip() for key in
               ("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked")}
        if any(value == "" for value in raw.values()):
            raise ValueError("Complete every field before requesting a prediction.")

        values = {
            "pclass": int(raw["pclass"]), "sex": int(raw["sex"]),
            "age": float(raw["age"]), "sibsp": int(raw["sibsp"]),
            "parch": int(raw["parch"]), "fare": float(raw["fare"]),
            "embarked": raw["embarked"].upper(),
        }
        if values["pclass"] not in (1, 2, 3):
            raise ValueError("Choose passenger class 1, 2, or 3.")
        if values["sex"] not in (0, 1):
            raise ValueError("Choose a passenger sex from the list.")
        if values["embarked"] not in ("S", "C", "Q"):
            raise ValueError("Choose a valid embarkation port.")
        if not np.isfinite(values["age"]) or not 0 <= values["age"] <= 100:
            raise ValueError("Age must be between 0 and 100.")
        if not np.isfinite(values["fare"]) or values["fare"] < 0:
            raise ValueError("Fare must be zero or greater.")
        if not 0 <= values["sibsp"] <= 10 or not 0 <= values["parch"] <= 10:
            raise ValueError("Family counts must be between 0 and 10.")

        # Titanic's commonly used label encoding: Southampton=0, Cherbourg=1,
        # Queenstown=2. Keep the model's training column names and order.
        feature_values = {
            "pclass": values["pclass"], "sex": values["sex"],
            "age": values["age"], "sibsp": values["sibsp"],
            "parch": values["parch"], "fare": values["fare"],
            "embarked": {"S": 0, "C": 1, "Q": 2}[values["embarked"]],
        }
        model_columns = list(getattr(model, "feature_names_in_",
                                     ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")))
        normalized_values = {key.lower(): value for key, value in feature_values.items()}
        try:
            row = [normalized_values[column.lower()] for column in model_columns]
        except KeyError as exc:
            raise RuntimeError(f"Unsuppsorted model feature: {exc.args[0]}") from exc
        features = pd.DataFrame([row], columns=model_columns)
        predicted = int(model.predict(features)[0])
        if predicted not in (0, 1):
            raise RuntimeError("The model returned an unexpected prediction.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(getattr(model, "classes_", range(len(probabilities))))
            survived_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        else:
            survived_probability = float(predicted)
        percentage = round(min(1.0, max(0.0, survived_probability)) * 100)
        return render_home(
            values=values, prediction=predicted,
            result="Survived" if predicted else "Did not survive",
            survived_probability=percentage,
            not_survived_probability=100 - percentage,
            confidence=percentage if predicted else 100 - percentage,
            family_size=values["sibsp"] + values["parch"] + 1,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return render_home(error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_home(error="The model could not make a prediction. Check the model file and try again."), 500


if __name__ == "__main__":
    app.run()
