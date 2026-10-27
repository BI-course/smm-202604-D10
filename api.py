"""
api.py -- Group D10 (BBT 4206) RESTish API serving four persisted artifacts:
  1. Regression      -- annual farm income (KES)
  2. Classification  -- invoice payment risk (On-Time / Late / Severely Late)
  3. Clustering      -- agricultural cooperative farmer segment
  4. ARM             -- food-delivery item recommendations
All artifacts are loaded ONCE from ./model at startup -- nothing is retrained here.
Run locally:  python api.py
"""
import glob
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost", "https://127.0.0.1",
    "http://localhost:63342", "null",
]}})

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Set to True ONLY if the regression notebook trained on np.log1p(target)
REGRESSION_TARGET_LOG1P =True


def load_artifact(prefix):
    matches = sorted(glob.glob(os.path.join(MODEL_DIR, f"{prefix}*.joblib")))
    if not matches:
        raise FileNotFoundError(f"No file starting with '{prefix}' in {MODEL_DIR}")
    return joblib.load(matches[0])


def feature_names(obj):
    return [str(f) for f in getattr(obj, "feature_names_in_", [])]


# ---------------- Load all four artifacts once ----------------
regression_pipeline = load_artifact("d10_farm_income")
classification_pipeline = load_artifact("d10_invoice_payment_risk")
clustering_artifact = load_artifact("d10_agricultural_cooperative_farmer")
arm_artifact = load_artifact("d10_food_delivery_arm")

REGRESSION_FEATURES = feature_names(regression_pipeline)
CLASSIFICATION_FEATURES = feature_names(classification_pipeline)
CLASS_LABELS = list(getattr(classification_pipeline, "classes_", classification_pipeline[-1].classes_))

# Clustering may be a Pipeline, a bare KMeans, or a dict like {"scaler":..., "kmeans":...}
clustering_scaler, cluster_names = None, {}
if isinstance(clustering_artifact, dict):
    clustering_model = next(v for v in clustering_artifact.values() if hasattr(v, "predict"))
    if not hasattr(clustering_model, "steps"):
        clustering_scaler = next((v for k, v in clustering_artifact.items()
                                  if "scal" in k.lower() and hasattr(v, "transform")), None)
    saved_feats = next((v for k, v in clustering_artifact.items() if "feature" in k.lower()), None)
    cluster_names = next((v for k, v in clustering_artifact.items()
                          if isinstance(v, dict) and ("label" in k.lower() or "name" in k.lower())), {})
else:
    clustering_model, saved_feats = clustering_artifact, None
CLUSTERING_FEATURES = (list(saved_feats) if saved_feats is not None
                       else feature_names(clustering_scaler) or feature_names(clustering_model))

recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]
NAME_TO_ID = {str(n).lower(): i for i, n in sku_lookup["product_name"].items()}

print("All four D10 artifacts loaded. The API is ready.")


# ---------------- Helpers ----------------
def build_row(features):
    """Validate the JSON body and return a one-row DataFrame in the model's column order."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, (jsonify({"error": "Request body must be a JSON object"}), 400)
    missing = [f for f in features if f not in payload]
    if missing:
        return None, (jsonify({"error": "Missing required fields", "missing": missing,
                               "required": features}), 400)
    row = pd.DataFrame([{f: payload[f] for f in features}])
    for c in row.columns:                      # JSON null -> NaN so the pipeline's imputer handles it
        if row[c].isna().all():
            row[c] = np.nan
    return row, None


def predict_safely(fn):
    try:
        return fn()
    except Exception as exc:                   # bad category, wrong type, etc.
        return jsonify({"error": "Prediction failed", "detail": str(exc)}), 422


# ---------------- Root: self-documenting index ----------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Group D10 model-serving API",
        "endpoints": {
            "POST /api/v1/models/farm-income-regressor/predictions": REGRESSION_FEATURES,
            "POST /api/v1/models/invoice-payment-risk-classifier/predictions": CLASSIFICATION_FEATURES,
            "POST /api/v1/models/farmer-segmenter/predictions": CLUSTERING_FEATURES,
            "POST /api/v1/models/food-delivery-recommender/predictions":
                {"cart": "list of item IDs or names", "top_n": "optional int (default 3)",
                 "items": {str(i): n for i, n in sku_lookup["product_name"].items()}},
        },
    })


# ---------------- 1. Regression ----------------
@app.route("/api/v1/models/farm-income-regressor/predictions", methods=["POST"])
def predict_farm_income():
    row, err = build_row(REGRESSION_FEATURES)
    if err:
        return err

    def run():
        pred = float(regression_pipeline.predict(row)[0])
        if REGRESSION_TARGET_LOG1P:
            pred = float(np.expm1(pred))
        return jsonify({"predicted_annual_farm_income_kes": round(pred, 2),
                        "model": "Linear Regression pipeline"})
    return predict_safely(run)


# ---------------- 2. Classification ----------------
@app.route("/api/v1/models/invoice-payment-risk-classifier/predictions", methods=["POST"])
def predict_invoice_risk():
    row, err = build_row(CLASSIFICATION_FEATURES)
    if err:
        return err

    def run():
        pred = classification_pipeline.predict(row)[0]
        probs = classification_pipeline.predict_proba(row)[0]
        return jsonify({"predicted_payment_risk": str(pred),
                        "class_probabilities": {str(l): round(float(p), 4)
                                                for l, p in zip(CLASS_LABELS, probs)},
                        "model": "Tuned Logistic Regression (class-weighted)"})
    return predict_safely(run)


# ---------------- 3. Clustering ----------------
@app.route("/api/v1/models/farmer-segmenter/predictions", methods=["POST"])
def predict_farmer_segment():
    row, err = build_row(CLUSTERING_FEATURES)
    if err:
        return err

    def run():
        X = clustering_scaler.transform(row) if clustering_scaler is not None else row
        cluster = int(clustering_model.predict(X)[0])
        return jsonify({"assigned_cluster": cluster,
                        "segment": cluster_names.get(cluster, f"Cluster {cluster}"),
                        "model": "K-Means"})
    return predict_safely(run)


# ---------------- 4. Association Rule Mining ----------------
@app.route("/api/v1/models/food-delivery-recommender/predictions", methods=["POST"])
def recommend_items():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload.get("cart"):
        return jsonify({"error": "Provide a non-empty 'cart' list, e.g. {\"cart\": [\"F01\", \"F02\"]}"}), 400
    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        return jsonify({"error": "'top_n' must be an integer"}), 400

    # accept item IDs (F01) or names (Beef Burger)
    cart = {NAME_TO_ID.get(str(i).lower(), i) for i in payload["cart"]}

    candidates = {}
    for antecedent, consequents in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for item, conf, lift in consequents:
                if item not in cart and (item not in candidates or conf > candidates[item][0]):
                    candidates[item] = (conf, lift)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
    return jsonify({
        "cart": sorted(map(str, cart)),
        "recommendations": [
            {"item": str(i),
             "product_name": str(sku_lookup.loc[i, "product_name"]) if i in sku_lookup.index else str(i),
             "confidence": round(conf, 3), "lift": round(lift, 2)}
            for i, (conf, lift) in ranked],
        "model": "Apriori association rules (precomputed lookup)",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)