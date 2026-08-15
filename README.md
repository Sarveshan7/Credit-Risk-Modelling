# Credit-Risk-Modelling
This project predicts credit card default risk using customer data. Traditional machine learning models such as logistic regression, support vector machine and random forest are compared to more advanced machine learning boosting models such as XGBoost, CatBoost and LightGBM. 

## Dataset
[Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) 
- Customer demographic, credit, and repayment history data from Taiwan, 2005.

# Approach
- Exploratory data analysis to understand default patterns across demographics, 
  credit limits, and repayment behaviour
- Data cleaning (consolidating rare categories in education/marriage fields)
- Addressed class imbalance in the target variable using SMOTE
- Hyperparameter tuning across all six models
- Evaluated models using ROC-AUC, given the class imbalance

# Results
Boosting models outperformed traditional approaches, with the best model achieving 
**77% ROC-AUC**.

# Tools
Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, LightGBM, Matplotlib, Seaborn
