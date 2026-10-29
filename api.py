"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- Annual farm income prediction
    2. Classification   -- Invoice payment risk category
    3. Clustering       -- Agricultural cooperative farmer segment assignment
    4. Association Rule -- Food recommendation from a cart
       Mining (ARM)

The regression and classification pipelines contain their trained preprocessing.
The classifier uses class weighting; its saved resampler is passthrough.
The clustering artifact contains a separate fitted preprocessing pipeline and
K-Means model, together with the expected feature order and segment names.
The recommender uses saved association rules and item information.
All artifacts are loaded once from disk; nothing is trained or mined here.

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "d10_farm_income_linear_regression_pipeline.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "d10_invoice_payment_risk_tuned_logistic_pipeline.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "d10_agricultural_cooperative_farmer_kmeans_clustering.joblib")
)
clustering_preprocessor = clustering_artifact["preprocessor"]
clustering_kmeans = clustering_artifact["kmeans"]
clustering_features = clustering_artifact["clustering_features"]
clustering_segment_names = clustering_artifact["segment_names"]
clustering_k = clustering_artifact["final_k"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "d10_food_delivery_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
item_lookup = arm_artifact["item_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# 1. REGRESSION -- predict annual farm income (KES)

@app.route("/api/v1/models/farm-income-regressor/predictions", methods=["POST"])
def predict_farm_income():
    payload = request.get_json()
    input_row = pd.DataFrame([payload])

    # The saved pipeline applies the trained preprocessing.
    predicted_log_income = regression_pipeline.predict(input_row)[0]

    # Convert the log-scale prediction back to Kenyan shillings.
    import numpy as np
    predicted_income_kes = float(np.expm1(predicted_log_income))

    return jsonify({
        "predicted_annual_farm_income_kes": round(predicted_income_kes, 2),
        "model": "Linear Regression",
    })


# 2. CLASSIFICATION -- predict invoice payment risk (On-Time/Late/Severely Late)

@app.route("/api/v1/models/invoice-payment-risk-classifier/predictions", methods=["POST"])
def predict_invoice_payment_risk():
    payload = request.get_json()
    input_row = pd.DataFrame([payload])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_invoice_payment_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Logistic Regression (tuned, class-weighted)",
    })


# 3. CLUSTERING -- assign a new farmer to an existing cooperative segment

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/farmer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"farm_size_acres": 40.0, "annual_yield_tonnes": 160.0,
#             "income_per_acre_kes": 85000.0, "years_farming_experience": 16.0,
#             "market_access_score": 55.0}'
#
# Apply the saved median imputer and scaler before predicting with K-Means.
# Preserve the numeric feature order recorded in the clustering artifact.
@app.route("/api/v1/models/farmer-segmenter/predictions", methods=["POST"])
def predict_farmer_segment():
    payload = request.get_json()
    input_row = pd.DataFrame([payload])[clustering_features]

    scaled_input = clustering_preprocessor.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "segment_name": clustering_segment_names[assigned_cluster],
        "model": f"K-Means (k={clustering_k}, standardized features)",
    })


# 4. ASSOCIATION RULE MINING -- recommend foods for a delivery cart

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/food-delivery-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["F10", "F11"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/food-delivery-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["F07", "F09"]}'
#
# Note: there is no scikit-learn ".predict()" here -- the mined association
# rules were precomputed into a lookup dictionary
# (antecedent food-item set -> ranked candidate dictionaries)
# during the notebook's persistence stage. This
# endpoint performs the exact same lookup+rank step a production recommender
# microservice would perform on every incoming request; it does not mine
# for the rules again.
@app.route("/api/v1/models/food-delivery-recommender/predictions", methods=["POST"])
def recommend_foods():
    payload = request.get_json()
    cart = set(payload.get("cart", []))
    top_n = int(payload.get("top_n", 3))

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for recommendation in consequent_list:
                item_id = recommendation["item_id"]
                if item_id not in cart:
                    existing = candidates.get(item_id)
                    if existing is None or (
                        recommendation["confidence"],
                        recommendation["lift"],
                        recommendation["support"],
                    ) > (
                        existing["confidence"],
                        existing["lift"],
                        existing["support"],
                    ):
                        candidates[item_id] = recommendation

    ranked = sorted(
        candidates.values(),
        key=lambda recommendation: (
            recommendation["confidence"],
            recommendation["lift"],
            recommendation["support"],
        ),
        reverse=True,
    )[:top_n]
    recommendations = [
        {
            "item_id": str(recommendation["item_id"]),
            "item_name": item_lookup.loc[recommendation["item_id"], "item_name"],
            "category": item_lookup.loc[recommendation["item_id"], "category"],
            "confidence": round(float(recommendation["confidence"]), 4),
            "lift": round(float(recommendation["lift"]), 4),
            "support": round(float(recommendation["support"]), 4),
        }
        for recommendation in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
