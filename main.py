import warnings

warnings.filterwarnings('ignore')  # supress warnings

# import modules
import numpy as np
import pandas as pd

# Show all columns of dataframe in pandas
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('employee_data_hr_comma_sep.xlsx')
# print(df.head())

# Data preprocessing (EDA)

print(df.shape)

# 1. Check for missing Values
print(df.info())  # get information about the data types in each column
print(df.isna().sum())  # get info on how many values are missing

# Output shows that data has no missing Values

# 2. Understand what factors contributed most to employee turnover

# 2.1 Heatmap of numerical values
# drop columns sales and salary


heat_data = df.drop(['sales', 'salary'], axis=1)

# Figure 1
plt.figure(figsize=(10, 8))  # create a new figure
sns.heatmap(data=heat_data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
# plt.show()  # renders only this figure

#### Inference from heatmap
# 1. Little correlation between any feature.
# 2. Highest negative correlation between satisfaction level and left.
# 3. Highest positive correlation between number of projects and average monthly work hours.

# Distribution Plot
plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
sns.distplot(df["satisfaction_level"])
plt.subplot(1, 3, 2)
sns.distplot(df["last_evaluation"])
plt.subplot(1, 3, 3)
sns.distplot(df["average_montly_hours"])
plt.tight_layout()
# plt.show()

# ### Inference from distribution plot.
# 1. Last evaluation and average monthly hours overlap in shape.
# 2. Satisfaction spikes has outlier in the non satisfaction peak.
# 3. Monthly hours peak between 160 and 270 hours. 270hours work/month means more than 12.5hours work/day.

#### 2.3 Bar Plot of Employee Project Count including the employees who left.

# print(df.head())
plt.figure(figsize=(5, 8))
sns.barplot(x=df["left"], y=df["number_project"], hue=df['left'])
plt.title(" Number of Projects: Employed vs. Left")
plt.tight_layout()
# plt.show()

# #### Inference from the plot:
# Zero means people still work and 1 means people left.
# 1. Little difference in project numbers between people who still work and people who left.
# 2. This plot does not tell anything about the satisfaction of the people who still work at the company.
# 3. 3.7 projects on average for people who still work at the company, might still be too many.

### 3 Clustering of Employees who left based on their satisfaction and evaluation

### 3.1 choose columns: satisfaction_level, last_eval and left
cluster_dat = df[["satisfaction_level", "last_evaluation", "left"]]
# print(cluster_dat.head())

# get only cases where people have left
left = cluster_dat[cluster_dat['left'] == 1]

#### 3.2 KMeans Clustering of employees who left the company into 3 clusters.
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(left)
labels = kmeans.fit_predict(left)
# print(labels)
left["clusters"] = labels
centroids = kmeans.cluster_centers_
# print(centroids)

plt.figure(figsize=(5, 8))
sns.scatterplot(x='satisfaction_level', y='last_evaluation', data=left, hue='clusters', palette='pastel')
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=25, c="black")
plt.title("KMeans Clustering - People Who Left")
# plt.show()

# #### Inference from the Clusters
# 1. Cluster 2, High Evaluation Score, but lowest satisfaction (Perhaps too many Projects, little Pay, too many work hours.)
# 2. Cluster 0, low evaluation score and medium satisfaction.
# 3. Cluster 1, high evaluation score, highest satisfaction. (Better offer elsewhere)
# 4. Of all the people who left, the ones with the highest Eval Score are highly trained and good at what they do.
# Likely the ones who are most unsatisfied, are overworked and also get paid too little.
# Employees who have the highest satisfaction level, could land a better job elsewhere.
# 5. To get a better idea, the Salary would need to be checked as well.



##################
### Compare salary and satisfaction among people who left
# since the salary column is categorical the label needs be transformed first
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# The label encoder is not a good choice here since it encodes alphabetically, which gives salary high a zero whereas medium
# is encode with 2 and low with a 1, this gives a really confusing plot.
# Therefore better with logical mapping
### 3.1 choose columns: satisfaction_level, last_eval and left
cluster_dat = df[["satisfaction_level", "salary",  "last_evaluation", "left"]]
# Define logical mapping
salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
# Apply mapping
cluster_dat['salary'] = cluster_dat['salary'].map(salary_mapping)

# print(cluster_dat.head())

# get only cases where people have left
left_salary = cluster_dat[cluster_dat['left'] == 1]

#### 3.2 KMeans Clustering of employees who left the company into 3 clusters.
kmeans = KMeans(n_clusters=3)
kmeans.fit(left_salary)
labels = kmeans.fit_predict(left_salary)
# print(labels)
left_salary["clusters"] = labels
centroids = kmeans.cluster_centers_
# print(centroids)

plt.figure(figsize=(5, 8))
sns.scatterplot(x='satisfaction_level', y='last_evaluation', data=left_salary, hue='salary', palette='pastel')
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=25, c="black")
plt.title("KMeans Clustering - People Who Left")
# plt.show()

### Read out from Plot.
# The centroids do not catch the clusters, likely because the salary is part of the dataset and not discrete range.
# Overall the people who left have low or medium salary.

###########
# Scaling the Non continues variable Salary which ranges from 0-2
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = cluster_dat.drop('left', axis=1)
y = cluster_dat['left']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add the 'left' column back
cluster_dat = X_scaled_df.copy()
cluster_dat['left'] = y.values

# print(cluster_dat.head())
# get only cases where people have left
left_salary_scaled= cluster_dat[cluster_dat['left'] == 1]

#### 3.2 KMeans Clustering of employees who left the company into 3 clusters.
kmeans = KMeans(n_clusters=3)
kmeans.fit(left_salary_scaled)
labels = kmeans.fit_predict(left_salary_scaled)
# print(labels)
left_salary_scaled["clusters"] = labels
centroids = kmeans.cluster_centers_
# print(centroids)

plt.figure(figsize=(5, 8))
sns.scatterplot(x='satisfaction_level', y='last_evaluation', data=left_salary_scaled, hue='salary', palette='pastel')
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=25, c="black")
plt.title("KMeans Clustering - People Who Left")
# plt.show()

### 4. Handle the left Class imbalance using SMOTE technique.

from sklearn.preprocessing import LabelEncoder
# separate data into numerical and categorical
numerical = df.drop(['sales', 'salary'], axis=1)
# print(numerical.head())

categorical = df[['sales', 'salary']]
# print(categorical.head())

le = LabelEncoder()
for var in categorical:
    categorical[var] = le.fit_transform(categorical[var])

# print(categorical.head())

# combined datasets again
df = pd.concat([numerical, categorical], axis = 1)
# print(df.head())

#### 4.2  do the stratified split into 80:20 with random state=123
new_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'sales', 'salary']

X = df[new_cols]
y = df['left']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



#### 4.3 Upsample the train data set
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train , y_train)

### 5 Model Building
#### 5.1 Logistic Regression Model with 5 fold CV plus Classification Report

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression()
model_logistic.fit(X_train , y_train)
model_logistic.score(X_train, y_train)


print(f"logistic regression model score: {model_logistic.score(X_test, y_test)}")


print(f"logistic regression cross validation score: {cross_val_score(model_logistic, X_test, y_test, cv=5).mean()}")

#### Classification Report Logistic Regression
from sklearn.metrics import classification_report
print(f"logistic regression cross val report: {classification_report(y_test, model_logistic.predict(X_test))}")

### output logistic regression model:
# logistic regression model score: 0.746
# logistic regression cross validation score: 0.763
# logistic regression cross val report:
#                   precision    recall  f1-score   support
#            0       0.91      0.74      0.82      2291
#            1       0.48      0.75      0.58       709
#
#     accuracy                           0.75      3000
#    macro avg       0.69      0.75      0.70      3000
# weighted avg       0.80      0.75      0.76      3000



#### 5.2 Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=10, max_depth=4)
model_rf.fit(X_train , y_train)
model_rf.score(X_train, y_train)

print(f"Random Forest Model Score: {model_rf.score(X_test, y_test)}")
# Random Forest Model Score: 0.9326666666666666

#####  Inference from Random Forest Model
# 1. probably over fitted, but pruned to 4 levels.

print(f"Random Forest Cross Val Score: {cross_val_score(model_rf, X_test, y_test, cv = 5).mean()}")
# Radnom Forest Cross Val Score: 0.938

#### Classification Report Random Forest
print(f"Random Forest Cross Classification Report: {classification_report(y_test, model_rf.predict(X_test))}")
# Random Forest Cross Classification Report:
#                   precision    recall  f1-score   support
#            0       0.98      0.93      0.95      2291
#            1       0.81      0.94      0.87       709
#
#     accuracy                           0.93      3000
#    macro avg       0.89      0.93      0.91      3000
# weighted avg       0.94      0.93      0.93      3000

#### 5.3 Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model_gb = GradientBoostingClassifier(n_estimators=100)
model_gb.fit(X_train , y_train)
model_gb.score(X_train, y_train)
print(f"Gradient Boosting Model Score: {model_gb.score(X_train, y_train)}")
# Gradient Boosting Model Score: 0.9582466892853234
print(f"Gradient Boosting Cross Validation Score: {cross_val_score(model_gb, X_test, y_test, cv=5).mean()}")
# Gradient Boosting Cross Validation Score: 0.9713333333333333
#### Classfication Report Gradient Boosting
print(f"Gradient Boosting Model Classification Report: {classification_report(y_test, model_gb.predict(X_test))}")
# Gradient Boosting Model Classification Report:
#                   precision    recall  f1-score   support
#            0       0.98      0.97      0.97      2291
#            1       0.90      0.94      0.92       709
#
#     accuracy                           0.96      3000
#    macro avg       0.94      0.95      0.94      3000
# weighted avg       0.96      0.96      0.96      3000


### 6 Identify the best model and justify the evaluation Metrics

#### 6.1 Find ROC/AUC Curve for each model and plot the ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#### Logistic Model ROC
logit_roc_auc = roc_auc_score(y_test, model_logistic.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model_logistic.predict_proba(X_test)[:,1])

#### Random Forest ROC
rf_roc_auc = roc_auc_score(y_test, model_rf.predict(X_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, model_rf.predict_proba(X_test)[:,1])

#### Gradient Boosting ROC
gb_roc_auc = roc_auc_score(y_test, model_gb.predict(X_test))
gb_fpr, gb_tpr, gb_thresholds = roc_curve(y_test, model_gb.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.3f)' % rf_roc_auc)
plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting (area = %0.3f)' % gb_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
# plt.show()

#### From the above AUC-ROC plot we can confirm that the best model to use is Gradient Boosting or Random Forest.

#### 6.2 Find the confusion Matrix for each model

from sklearn.metrics import  confusion_matrix
#### confusion matrix logistic regression model
print(f"Confusion Matrix Logistic Regression: {confusion_matrix(y_test, model_logistic.predict(X_test))}")
# Confusion Matrix Logistic Regression:
# [[1702  589]
#  [ 174  535]]
cm_lr = confusion_matrix(y_test, model_logistic.predict(X_test))
# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed','Left'], yticklabels=['Stayed','Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Logistic Regression')
# plt.show()

#### confusion matrix Random Forest Model
print(f"Confusion Matrix Random Forest: {confusion_matrix(y_test, model_rf.predict(X_test))}")
# Confusion Matrix Random Forest:
# [[2202   89]
#  [  52  657]]
cm_rf= confusion_matrix(y_test, model_rf.predict(X_test))
# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed','Left'], yticklabels=['Stayed','Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Random Forest')
# plt.show()


#### confusion matrix Gradient Boost Model
print(f"Confusion Matrix Gradient Boost: {confusion_matrix(y_test, model_gb.predict(X_test))}")
# Confusion Matrix Gradient Boost:
# [[2215   76]
#  [  45  664]]
cm_gb= confusion_matrix(y_test, model_gb.predict(X_test))
# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed','Left'], yticklabels=['Stayed','Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Gradient Boost')
# plt.show()



# plot all models confusion matrices into one figure:

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

models = {
    "Logistic Regression": model_logistic,
    "Random Forest": model_rf,
    "Gradient Boost": model_gb
}

plt.figure(figsize=(18,5))

for i, (name, model) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed','Left'], yticklabels=['Stayed','Left'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()


#### Inference from Confusion matrix
# 1. While Random Forest and Gradient Boosting have both a high precision to identify true pos and true negative,
# the false positives and false negatives can better be assessed by RECALL.

### 7 Suggest Retention Strategies for targeted employees.
#### 7.1 Using the best model predict the probability of employee turnover in th test data.
# 1. in this case the gradient boosting model

y_prob = model_gb.predict_proba(X_test)
# print(y_prob)

turnover_prob = y_prob[:, 1]
# print(turnover_prob)

safe_zone_threshold = 0.20  # 20%
low_risk_zone_threshold = 0.40  # 40%
moderate_risk_zone_threshold = 0.60  # 60%

# High risk is anything above moderate risk

# Categorize employees based on probability scores
def categorize_employee(probability_score):
    if probability_score < safe_zone_threshold:
        return "Safe Zone (Green)"
    elif probability_score < low_risk_zone_threshold:
        return "Low Risk Zone (Yellow)"
    elif probability_score < moderate_risk_zone_threshold:
        return "Moderate Risk Zone (Orange)"
    else:
        return "High Risk Zone (Red)"

# Categorize employees in the test set
employee_categories = [categorize_employee(score) for score in turnover_prob]

# Print the categorized employees
for i, category in enumerate(employee_categories):
    print(f"Employee {i+1}: {category}")

# ### Retention Strategies:
# 1. employees should be given less Working Hours
# 2. Number of projects should be no more than 4
# 3. Employee salary should be increased