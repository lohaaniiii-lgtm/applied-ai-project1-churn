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
