Employee Turnover Prediction — Machine Learning Project
This project explores employee turnover using real-world HR data and applies machine-learning models to predict which employees are most likely to leave a company.
It includes Exploratory Data Analysis (EDA), clustering, class imbalance handling, and performance comparison of three classification models: Logistic Regression, Random Forest, and Gradient Boosting.
📊 Project Overview
Companies face high costs when losing trained employees. The goal of this project is to:
Understand what factors contribute to turnover
Identify behavior patterns among employees who left
Build ML models that accurately predict employee churn
Use prediction probabilities to categorize employees into actionable risk groups
Dataset source: Modified from Kaggle — HR Analytics: Employee Turnover.
🔍 Exploratory Data Analysis (EDA)
Key findings from EDA:
Satisfaction level shows the strongest negative correlation with turnover.
Employees with high evaluation scores but low satisfaction frequently left—suggesting overwork or insufficient compensation.
Monthly working hours peak between 160–270 hours, indicating potential workload strains.
Number of projects was not a strong standalone predictor.
Visualizations included:
Correlation heatmap
Distribution plots
Bar plot of project count by turnover status
🧭 Clustering Analysis
K-Means was applied to employees who left to understand different patterns among them:
Cluster 2: High evaluation + low satisfaction → likely overworked high performers
Cluster 1: Medium satisfaction + low evaluation → possibly disengaged
Cluster 2: High satisfaction + high evaluation → may have found better opportunities
Salary was later encoded and scaled for consistency during clustering.
⚖️ Handling Imbalanced Data
Because only a minority of employees leave, the dataset was imbalanced.
To address this, SMOTE (Synthetic Minority Oversampling Technique) was used to upsample the minority class in the training data, ensuring fair model training.
🤖 Models Trained
Three classification models were built and compared using cross-validation, ROC/AUC, and confusion matrices:
1. Logistic Regression
Accuracy: ~74%
Strong for baseline interpretation
Lower recall for employees who left
2. Random Forest
Accuracy: ~93%
Strong precision and recall
Good at capturing nonlinear relationships
3. Gradient Boosting (Best Model)
Accuracy: ~96%
Highest recall and f1-score for the “left” class
Best overall model for identifying at-risk employees
📈 Why Evaluation Metrics Matter
Accuracy alone is misleading because “left” cases are rare.
A model might achieve high accuracy simply by predicting that everyone stays.
Key metrics used:
Precision – Of predicted leavers, how many truly left?
Recall – How many of all true leavers the model successfully identified.
F1-score – Balance between precision and recall
ROC/AUC – Overall model discriminative ability
Recall is emphasized, because failing to identify a true leaver is costlier for HR than flagging a possible false alarm.
🚦 Turnover Risk Categorization
Using the best model (Gradient Boosting), employees were classified into:
Safe Zone (<20%)
Low Risk (20–40%)
Moderate Risk (40–60%)
High Risk (>60%)
This helps HR teams intervene early.
🧩 Retention Strategy Suggestions
Based on model insights:
Reduce excessive working hours
Limit the number of simultaneous projects
Improve compensation for high-performing employees at risk of burnout
🗂️ Repository Contents
main.py — Project notebook
employee_data_hr_comma_sep.xlsx — Dataset (if included)
Python scripts for preprocessing, modeling, and evaluation
Visualizations and model evaluation outputs
