<div align="center">

![Car Price Prediction banner](images/project_banner.png)

# Used Car Price Prediction Using Machine Learning

**End-to-end exploratory analysis, preprocessing pipelines, and regression model comparison on used-car listing data.**

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

</div>

## Overview

This project explores a used-car listings dataset and trains machine learning models to estimate **`Price`** using specifications such as make, model, year, mileage, transmission, owner history, body dimensions, and derived engine-performance features.

The main work is in [`Project_Final.ipynb`](Project_Final.ipynb), with the source data in [`car_details.csv`](car_details.csv). The figures in this README are generated from the uploaded dataset; the model-comparison chart uses the **saved notebook outputs**, not a fresh retraining run.

## Highlights

| Item | Details |
| --- | --- |
| Dataset | 2,059 rows × 20 original columns |
| Prediction target | `Price` (continuous numeric value) |
| Data split | 80% training / 20% testing (`random_state=42`) |
| Best model in saved run | **Random Forest Regressor** |
| Random Forest test R² | **0.8159** |
| Random Forest test RMSE | **1,134,247** (dataset price units) |
| Regressors compared | Linear, Random Forest, Ridge, Lasso, Elastic Net, SGD |

> **Metric note:** An R² of 0.8159 is **not** 81.59% classification accuracy. R² describes the proportion of test-target variance explained relative to a mean-value baseline. The uploaded CSV does not explicitly document the price currency, so this README reports error in **dataset price units**.

## Visual Summary

![Model evaluation chart](images/model_comparison.png)

| Price distribution | Common car makes |
| :---: | :---: |
| ![Price distribution](images/eda_price_distribution.png) | ![Common car makes](images/eda_top_makes.png) |

## Dataset

- **Rows:** 2,059
- **Original columns:** 20
- **Target:** `Price`
- **Example predictor columns:** `Make`, `Model`, `Year`, `Kilometer`, `Fuel Type`, `Transmission`, `Location`, `Owner`, `Seller Type`, `Engine`, `Max Power`, `Max Torque`, `Length`, `Width`, `Height`.
- **Exact duplicate rows in uploaded CSV:** 0
- **Missing values:** Several fields are incomplete, especially `Drivetrain`, `Fuel Tank Capacity`, `Engine`, `Max Power`, `Max Torque` and car dimensions. Missing values are handled by fitted preprocessing pipelines rather than dropping all affected rows.

The notebook inspects numerical distributions, IQR-based potential outliers, missing values and categorical frequencies. The original price-distribution visualization hides the top 1% of prices **for display only**; the source data is not trimmed in the training split.

![Missing values](images/eda_missing_values.png)

> **Dataset provenance:** The dataset URL and license were not supplied. Add the original provider's attribution/link when known and confirm you may redistribute `car_details.csv` before publishing publicly.

## Workflow

![ML workflow](images/project_workflow.png)

1. **Load and inspect** the dataset using Pandas (`shape`, `head`, `describe`, `info`, missing-value checks).
2. **Explore distributions and relationships** with histograms, scatterplots, boxplots, median-price comparisons, and a correlation heatmap.
3. **Engineer numerical features** from strings such as maximum power and torque (BHP, RPM, Nm). The notebook separately calculates `Engine Size(cc)` in `df` for exploration; **this engineered column is not included in the `X` used to train models**.
4. **Build a `ColumnTransformer`** using median imputation and scaling for numeric features, frequent-value imputation and ordinal encoding for owner history, and frequent-value imputation and one-hot encoding for other categorical features.
5. **Split 80/20**, train a selection of regressors, and evaluate with MSE, RMSE, and R².

| Numeric correlation | Median prices by transmission |
| :---: | :---: |
| ![Numerical feature correlations](images/eda_correlations.png) | ![Transmission vs median price](images/eda_transmission_medians.png) |

## Models and Results

The following numbers were extracted from outputs saved in the **uploaded notebook** and are also recorded in [`results/model_metrics.csv`](results/model_metrics.csv).

| Regression model | Train R² | Test R² ↑ | Test RMSE ↓ |
| --- | ---: | ---: | ---: |
| **Random Forest** | **0.9780** | **0.8159** | **1,134,247** |
| Ridge Regression | 0.9476 | 0.7526 | 1,314,622 |
| SGD Regression | 0.7427 | 0.7445 | 1,336,103 |
| Elastic Net | 0.6344 | 0.6005 | 1,670,585 |
| Lasso Regression | 0.9923 | 0.5606 | 1,752,029 |
| Linear Regression | 0.9925 | 0.0781 | 2,537,855 |

**Interpretation:** Random Forest has the strongest reported test R² and lowest test RMSE among the six regressors. Its training R² is higher than its testing R², which indicates a generalization gap. Reported results can vary between runs because the notebook does not fix the Random Forest or SGD estimator random seeds.

### Important modeling notes

- The notebook also fits **Logistic Regression** to the continuous price target. This is a classification experiment, **not an appropriate standard method for continuous price regression**, and its accuracy should not be compared with the regression models' R² values.
- The notebook's `accuracy_score(y_test, y_predict.round())` for **SGD Regression** is a classification-style calculation and should **not** be used as a regression performance metric. Use **R², MAE, MSE, or RMSE** instead.
- The existing **Lasso** training emitted a convergence warning. Its result should be interpreted with that limitation.
- The notebook is preserved as supplied; the documentation does not claim these modeling limitations have been fixed.

## Run Locally

### 1. Clone this repository

Replace `<YOUR_USERNAME>` with your GitHub username:

```bash
git clone https://github.com/<YOUR_USERNAME>/Car-Price-Prediction-Using-Machine-Learning.git
cd Car-Price-Prediction-Using-Machine-Learning
```

Alternatively, download the repository ZIP and extract it.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

On **macOS/Linux**:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Open the Jupyter notebook

```bash
jupyter notebook Project_Final.ipynb
```

Run cells in order. Some training stages may take time on ordinary hardware; model scores may differ from the saved outputs due to versions or estimator randomness. `car_details.csv` must remain alongside the notebook because it is loaded with a relative file path.

### Regenerate the README charts

```bash
python scripts/generate_figures.py
```

This reads `car_details.csv` and `results/model_metrics.csv` and refreshes the PNGs in `images/`. It **does not retrain the models**.

## Repository Layout

```text
Car-Price-Prediction-Using-Machine-Learning/
├── Project_Final.ipynb          # Original notebook
├── car_details.csv             # Dataset (verify redistribution rights)
├── README.md                   # This documentation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excluded local files/secrets
├── repository_description.txt  # Description and GitHub topics
├── results/
│   └── model_metrics.csv       # Saved notebook result values
├── scripts/
│   └── generate_figures.py     # Recreate the README charts
└── images/
    ├── project_banner.png
    ├── project_workflow.png
    ├── model_comparison.png
    ├── eda_price_distribution.png
    ├── eda_top_makes.png
    ├── eda_missing_values.png
    ├── eda_correlations.png
    └── eda_transmission_medians.png
```

## Potential Improvements

- Exclude Logistic Regression from continuous-price model comparisons.
- Set explicit seeds for stochastic regressors for more reproducible benchmarks.
- Add cross-validation and hyperparameter tuning, evaluated without test-set leakage.
- Compare against a median-price baseline and add MAE to the model report.
- Review high-cardinality categorical features such as `Model` for potential overfitting.
- Investigate the cause of Lasso convergence warnings and optimize preprocessing/regularization.
- Add `Engine Size(cc)` to `X` if its inclusion is validated and improves held-out performance.

## Project Status

**Educational machine learning project.** Reported metrics are observations from the saved notebook outputs, not a guarantee of future prediction quality or production readiness.
