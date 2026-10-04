# Customer Churn Prediction

## Week 1: Exploratory Data Analysis

Notebook: `week1-eda.ipynb`

### Dataset
- Source: Telco Customer Churn (Kaggle)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

### Key Findings
- 26.54% of customers churned (1,869 out of 7,043)
- Month-to-month contracts churn at ~42%, vs ~11% for one-year and ~3% for two-year contracts
- Tenure is the strongest numeric predictor of churn (correlation of -0.35)
- Fiber optic customers churn at ~42%, far higher than DSL (~19%) or no internet service (~7%)
- Electronic check users churn at ~45%, nearly 3x the rate of autopay users (bank transfer/credit card)

### Setup
Open the Kaggle notebook or run locally:
```
pip install pandas numpy matplotlib seaborn
```

## Week 2: Building ML Models

Notebook: `week2-ml-models.ipynb`

- Baseline (always "stay"): accuracy 73.46%
- Best model: Logistic Regression, AUC 0.842 (tied with Random Forest), recall 0.921 at threshold 0.15
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.15, because a missed churner (PKR 6,000) costs 6x more than an unnecessary retention offer (PKR 1,000), so the cost-optimal threshold favors recall
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on Random Forest AUC: 0.8422 -> 0.8420 (no improvement, Random Forest likely already captures these interactions through combinations of raw features)
- Biggest lesson: The best model isn't the one with the highest accuracy. It's the one whose threshold and mistakes actually match what the business can afford, and Logistic Regression's interpretability made that decision far easier to defend than a black-box model would have.

## Week 3: Model Optimization and Unsupervised Learning

Notebook: `week3-optimization.ipynb`

- Split-to-split accuracy range across 20 seeds: 0.780 to 0.828 (std 0.0104, theoretical SE 0.0107)
- 5-fold CV AUC: LR 0.846 +/- 0.013, RF 0.844 +/- 0.011, XGBoost 0.850 +/- 0.012 (statistically tied)
- Tuning: best RF params max_depth=8, max_features='sqrt', min_samples_leaf=20; grid vs random search time 103 s vs 114 s (24 settings each)
- Final model: tuned XGBoost, test AUC 0.8483 (used once), vs 0.842 AUC for my Week 2 best model (Logistic Regression); the gain is small and within CV noise, since XGBoost, LR and RF are statistically tied. Early stopping chose 247 trees
- Customer segments (k = 4): Mid-tenure high spend (at risk) 43% churn, New low spend 32%, Loyal high spend bundled 14%, Long-tenure low spend 5%
- PCA: 15 of 30 components explain 90% of the variance
- Biggest lesson: One score from one split can fool you. The same model gave me 78% to 83% just by changing the split, so now I trust the average of 5-fold CV with its std.
