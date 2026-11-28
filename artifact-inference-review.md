# Artifact-to-inference review: D10 API

- **Author:** Ibrahim Omoi
- **Branch:** `ibrahim-artifact-review` (from `origin/wayne-practice`)
- **Date:** 2026-11-28
- **API commit tested:** `53ff5d9c9fcaef73683e45bef2c9b334022614c5` (from `git rev-parse HEAD`, taken before committing this report)
- **Repository:** https://github.com/BI-course/smm-202604-D10

Code links below point to `api.py` at the tested commit.

## 1. Saved artifacts exist

`dir model` shows eight `.joblib` files. The four used by the D10 API are:

| File in `model/` | Size (bytes) |
|---|---|
| `d10_farm_income_linear_regression_pipeline.joblib` | 5,546 |
| `d10_invoice_payment_risk_tuned_logistic_pipeline.joblib` | 9,142 |
| `d10_agricultural_cooperative_farmer_kmeans_clustering.joblib` | 5,044 |
| `d10_food_delivery_recommendation_rules.joblib` | 11,868 |

The other four files (`apriori_recommendation_rules`, `kmeans_mall_customer_segmentation`, `lasso_regressor_for_sme_revenue`, `svc_classifier_for_sme_credit_risk`) are from earlier labs. `Select-String -Path api.py -Pattern "load\(|\.joblib"` finds exactly four loaders (lines 72, 76, 80, 89), so none of those four is loaded by the API.

## 2. Route-by-route trace

| Route | Saved file | Loaded at | Route function | What happens between input and output |
|---|---|---|---|---|
| `farm-income-regressor` | `d10_farm_income_linear_regression_pipeline.joblib` | [`api.py:72`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L72) as `regression_pipeline` | `predict_farm_income()` ([`api.py:101`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L101)) | The JSON body becomes a one-row DataFrame (102-103). `regression_pipeline.predict` applies the saved preprocessing and the linear model and returns a **log-scale** income (106). `np.expm1` converts it back to **KES** (110). The result is rounded to 2 decimals in the response (113). |
| `invoice-payment-risk-classifier` | `d10_invoice_payment_risk_tuned_logistic_pipeline.joblib` | [`api.py:76`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L76) as `classification_pipeline` | `predict_invoice_payment_risk()` ([`api.py:121`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L121)) | One-row DataFrame (122-123). `predict` returns the class (125). `predict_proba` returns one probability per class (126). The class order comes from `named_steps["model"].classes_` (127). The response gives the predicted category plus every class's probability rounded to 4 decimals (130-134). |
| `farmer-segmenter` | `d10_agricultural_cooperative_farmer_kmeans_clustering.joblib` | [`api.py:80`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L80) as `clustering_artifact`, a dictionary unpacked at 83-87 | `predict_farmer_segment()` ([`api.py:151`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L151)) | The DataFrame is cut to the saved `clustering_features`, which fixes the column order (153). `clustering_preprocessor.transform` applies the saved imputer and scaler (155; the code comment at 148-149 names a median imputer and scaler). `clustering_kmeans.predict` returns the cluster number (156). The label comes from `clustering_segment_names[assigned_cluster]` (160). |
| `food-delivery-recommender` | `d10_food_delivery_recommendation_rules.joblib` | [`api.py:89`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L89) as `arm_artifact`, unpacked at 92-93 into `recommendation_lookup` and `item_lookup` | `recommend_foods()` ([`api.py:184`](https://github.com/BI-course/smm-202604-D10/blob/53ff5d9c9fcaef73683e45bef2c9b334022614c5/api.py#L184)) | The cart becomes a set and `top_n` defaults to 3 (186-187). The loop at 190-205 visits each stored antecedent set; if it is a subset of the cart, its consequent items that are not already in the cart become candidates, keeping the best `(confidence, lift, support)` per item. Candidates are sorted by that same tuple, highest first, and cut to `top_n` (207-215). Item names and categories come from `item_lookup` (219-220). |

## 3. Finding: where loading and inference happen

- **Loading:** all four artifacts are loaded once, at module level, when the server starts: lines 72-93, followed by the "All the four artifacts have been loaded" message at line 95. The comment at lines 63-69 explains the design: loading inside a route would re-read the file on every request.
- **Inference:** only inside the four route functions: `regression_pipeline.predict` (106), `classification_pipeline.predict` / `predict_proba` (125-126), `clustering_preprocessor.transform` / `clustering_kmeans.predict` (155-156), and the lookup-and-rank loop (190-215).
- **Training check:** I ran

  ```powershell
  Select-String -Path api.py -Pattern "\.fit|apriori|association_rules"
  ```

  The only match was line 231, `"model": "Apriori association rules, cleaned + precomputed lookup"`. That is a text label in the JSON response, not a call. There is no `.fit`, `.fit_transform`, `apriori(...)` or `association_rules(...)` anywhere in `api.py`.

**Result:** No route fits a model or mines new rules; all four only load saved artifacts and predict. The recommender is not a scikit-learn model at all: it looks up precomputed rules and ranks them (see the comment at lines 176-182).

## 4. Local request and response

Command run (API started in terminal 1 with `.\.venv\Scripts\python.exe api.py`):

```powershell
TO FILL IN: paste the exact command you ran
```

Response (status and body):

```text
TO FILL IN: paste the real output
```

Expected from the reference outputs: status 200, `assigned_cluster` 1, `segment_name` "Established Commercial High-Yield Farmers". The `assigned_cluster` and `segment_name` fields confirm the path traced in section 2: column selection (153), preprocessor (155), K-Means predict (156), label lookup (160).

## 5. What I learned

TO FILL IN: one sentence in your own words. Starting point: the API trains nothing; it loads files saved in earlier labs once at startup and only runs them on each request, which keeps responses fast and repeatable but means the saved models must match the installed library versions.

## 6. Failures and retests

- The first `git clone` failed with `Empty reply from server` (a network problem). I ran the same command again without changes and it succeeded (609 objects).
- Local request: TO FILL IN (write "no failures" if it worked first time).
