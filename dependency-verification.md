# Dependency verification: D10 API

**Author:** Melvin Muriithi (152384)
**Date:** 28 November 2026
**Branch:** `melvin-compatibility-check` (created from `origin/wayne-practice`)
**API commit tested:** `53ff5d9c9fcaef73683e45bef2c9b334022614c5` ("Serve D10 models and pin compatible dependencies")

## 1. My earlier API and dependency work

| Branch | Commits | What it contains |
|---|---|---|
| [`melvin-api`](https://github.com/BI-course/smm-202604-D10/tree/melvin-api) | `f3f1218`, `4bc4841` | First D10 version of `api.py`, the four D10 `.joblib` artifacts, and the dependency pins. This branch is deployed to Render at https://smm-202604-d10-melvin.onrender.com. |
| [`melvin-api-pr`](https://github.com/BI-course/smm-202604-D10/tree/melvin-api-pr) | `ed4ad1c`, `8870ccb` | The same work rebased onto `main` for review in [PR #2](https://github.com/BI-course/smm-202604-D10/pull/2). |

**API changes I made:**
- Replaced the lecturer's demo routes with four D10 routes: farm income regressor, invoice payment-risk classifier, farmer segmenter and food-delivery recommender.
- Loaded each artifact once at startup with `joblib.load`. No route trains a model.
- Added input validation. A missing field returns HTTP 400 with the list of missing and required fields. A bad value returns HTTP 422.
- Added a `GET /` index that lists each endpoint's required fields.
- Converted the regression output back to KES with `np.expm1`, because the model was trained on `log1p(annual_farm_income_kes)`.
- Exported the food-delivery Apriori rules from Colab into the `recommendation_lookup` / `sku_lookup` format that the API expects.

**Dependency changes I made, and why:**
1. **`scikit-learn==1.6.1` instead of 1.9.0.** The group's models were trained and saved in Google Colab with scikit-learn 1.6.1. Under 1.9.0, `api.py` failed at startup with `AttributeError: Can't get attribute '_RemainderColsList' on sklearn.compose._column_transformer`. That class exists in 1.6.1's `ColumnTransformer` but not in 1.9.0, so the pickled pipeline could not be rebuilt.
2. **`mlxtend==0.23.4` instead of 0.25.0.** mlxtend 0.25.0 requires `scikit-learn>=1.8.0`, which conflicts with the 1.6.1 pin. The Docker build failed with `ResolutionImpossible`. The API does not call mlxtend at runtime, because the rules are a precomputed lookup, so the older compatible version is safe.
3. **`imbalanced-learn==0.13.0`.** I pinned this on my branch to match the Colab environment. Wayne's branch uses 0.14.2 instead, which is also compatible (see below).

## 2. Constraints on `wayne-practice`

`requirements/constraints.txt` at `53ff5d9`:

```
imbalanced-learn==0.14.2
joblib==1.6.0
mlxtend==0.23.4
scikit-learn==1.6.1
```

Review:
- **scikit-learn 1.6.1:** matches the version that saved all four artifacts. Correct.
- **mlxtend 0.23.4:** compatible with scikit-learn 1.6.1. Correct.
- **imbalanced-learn 0.14.2:** differs from my 0.13.0 pin. pip accepted it alongside scikit-learn 1.6.1, and the classifier pipeline loaded and predicted correctly (section 3). No change needed.
- **joblib 1.6.0:** fine.
- **Unpinned packages:** numpy, pandas, scipy and flask are not pinned. The resolver installed numpy 2.5.5, pandas 3.1.1 and scipy 1.18.1, and everything worked. A future numpy or pandas release could still change behaviour, so pinning them would make Render and Docker builds fully reproducible.
- **`requirements/prod.txt`:** pulls `base.txt` from the lecturer's upstream repository URL, not our local file. The two package lists are currently the same, and the Docker build installs our local `base.txt` first. Pointing it at `-r base.txt` would remove the outside dependency.

## 3. Checks run

**Environment:** fresh Python 3.12.3 virtual environment on Linux, at commit `53ff5d9`.

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/base.txt -c requirements/constraints.txt
```

This installed: scikit-learn 1.6.1, imbalanced-learn 0.14.2, mlxtend 0.23.4, joblib 1.6.0, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, Flask 3.1.3, flask-cors 6.0.5.

### Check 1: `pip check`

```
$ .venv/bin/python -m pip check
No broken requirements found.
```

Exit code 0. **Pass.**

### Check 2: import `api` with scikit-learn version mismatches treated as errors

```
$ .venv/bin/python -c "import warnings; from sklearn.exceptions import InconsistentVersionWarning; warnings.simplefilter('error', InconsistentVersionWarning); import sklearn, mlxtend; print('scikit-learn', sklearn.__version__, 'mlxtend', mlxtend.__version__); import api"
scikit-learn 1.6.1 mlxtend 0.23.4
All the four artifacts have been loaded. The API is ready.
```

Exit code 0. No `InconsistentVersionWarning` was raised, which confirms all four artifacts were saved with the same scikit-learn version that is installed. **Pass.**

### Check 3: the four shared requests, run in-process

With the same warning filter active, I sent the four shared sample payloads to the routes using Flask's test client.

| Endpoint | Status | Result | Matches reference |
|---|---|---|---|
| `farm-income-regressor` | 200 | KES 357,577.40 | Yes |
| `invoice-payment-risk-classifier` | 200 | Severely Late, probability 0.9808 | Yes |
| `farmer-segmenter` | 200 | Cluster 1, Established Commercial High-Yield Farmers | Yes |
| `food-delivery-recommender` | 200 | F08 (Miso Soup), confidence 0.8782 | Yes |

## 4. Why the saved models and scikit-learn must be compatible

A `.joblib` file is a pickle. It stores each fitted object's internal attributes and the names of the classes it was built from, not the source code. When the API loads it, Python rebuilds those objects from whatever scikit-learn is installed.

If that version is different:
- The internal class layout may have changed. A class or attribute can be missing, so loading fails (as with `_RemainderColsList` under 1.9.0).
- The model may load but compute differently, with only an `InconsistentVersionWarning`. That risks silently wrong predictions.

Pinning `scikit-learn==1.6.1` in `constraints.txt` keeps the training environment (Colab), local Flask, Docker and Render on one version. Treating `InconsistentVersionWarning` as an error turns a silent risk into a visible failure.

## 5. Failures and follow-ups

- **Failures on the current `wayne-practice` commit:** none.
- **Failures I found and fixed earlier on `melvin-api`:**
  - the `_RemainderColsList` load error under scikit-learn 1.9.0, fixed by pinning 1.6.1;
  - the mlxtend/scikit-learn `ResolutionImpossible` error in the Docker build, fixed by pinning mlxtend 0.23.4.
- **Suggestions for Wayne (optional):**
  - pin numpy, pandas and scipy to the versions above;
  - change `prod.txt` to `-r base.txt`.

## What I learned

A trained model is only reproducible together with the exact library versions that saved it. Making dependency mismatches fail loudly at startup is much safer than letting them change predictions silently.
