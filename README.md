# Credit Risk Modelling

Predicting whether a credit card customer will default on their payment next month, comparing traditional machine learning models (Logistic Regression, SVM, Random Forest) against gradient boosting models (XGBoost, CatBoost, LightGBM).

Originally built as my university final year project; the code here is the full modelling pipeline, cleaned up for a portfolio.

## Dataset

[Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI ML Repository, via Kaggle): 30,000 credit card customers in Taiwan, April to September 2005. 25 variables covering demographics, credit limit, 6 months of repayment status, bill amounts, and payment amounts.

## Approach

1. **EDA & cleaning**: consolidated undocumented category codes in `EDUCATION`/`MARRIAGE`, checked for duplicates/nulls (none found), examined default patterns across demographics, credit limit, age, and repayment history
2. **Baseline models**: trained all six models with default hyperparameters as a reference point
3. **Hyperparameter tuning**: `RandomizedSearchCV`/`GridSearchCV` with 5-fold cross-validation, optimising for ROC-AUC
4. **Class imbalance**: the target is ~78% non-default / 22% default, so accuracy alone is misleading. Applied SMOTE to the training data and compared results with/without it
5. **Feature importance**: extracted and compared top features across all models, before and after SMOTE

## Results

5-fold CV ROC-AUC (tuned models):

| Model | Without SMOTE | With SMOTE |
|---|---|---|
| Logistic Regression | 0.723 | 0.722 |
| Random Forest | 0.780 | 0.780 |
| XGBoost | 0.780 | 0.784 |
| CatBoost | 0.785 | 0.784 |
| LightGBM | 0.779 | 0.784 |

**Boosting models (XGBoost, CatBoost, LightGBM) and Random Forest all land around 77-78% ROC-AUC**, well ahead of Logistic Regression and SVM.

SMOTE has almost no effect on ROC-AUC, but it meaningfully **improves recall on defaulters** (roughly 0.33-0.36 up to 0.51-0.57) at the cost of precision on that class. For a credit risk problem, missing an actual defaulter is generally costlier than a false alarm, so I'd favour the SMOTE-balanced models despite the precision trade-off.

The most recent repayment status (`PAY_0`) is consistently the strongest predictor of default across every model.

## Tools

Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, LightGBM, imbalanced-learn (SMOTE), Matplotlib, Seaborn

## Notes / limitations

1. Precision and recall on the default class still sit around 0.5 even after tuning. There's real room to improve here, likely through feature engineering (e.g. interaction terms) rather than more hyperparameter search.
2. Hyperparameter search was capped by compute/time (`RandomizedSearchCV` with a limited iteration budget, not an exhaustive grid).
3. Single 6-month snapshot from Taiwan, 2005. Patterns here wouldn't necessarily generalise to other markets or time periods without retraining.
