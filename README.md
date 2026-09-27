# Customer Churn Prediction

## Week 1: Exploratory Data Analysis

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
- Baseline (always "stay"): accuracy 73.46%
- Best model: Logistic Regression, AUC 0.842, recall 0.921 at threshold 0.15
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.15, because a missed churner (PKR 6,000) costs 6x more than an 
  unnecessary retention offer (PKR 1,000), so the cost-optimal threshold favors recall
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 
  0.8422 -> 0.8420 (no improvement — Random Forest likely already captures these 
  interactions through combinations of raw features)
- Biggest lesson:  The best model isn't the one with the highest accuracy — it's the one whose 
  threshold and mistakes actually match what the business can afford, and Logistic Regression's 
  interpretability made that decision far easier to defend than a black-box model would have.
