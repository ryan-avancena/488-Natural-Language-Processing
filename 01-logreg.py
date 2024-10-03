import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

"""
The data is related with direct marketing campaigns of a Portuguese banking institution. 
The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, 
    in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

"""

df = pd.read_csv('./datasets/marketing_data.csv',sep=';')

""" DATA OBSERVATION """

print(f'numeric: {df.select_dtypes(include="number").columns}')
print(f'categorical: {df.select_dtypes(include="object").columns}')
# print(f'nulls: {df.isnull().sum()}')
print(f'y counts: {df["y"].value_counts()}\n')
# print(f'default counts: {df["default"].value_counts()}\n')
# print(f'month counts: {df["month"].value_counts()}\n')
# print(f'day counts: {df["day"].value_counts()}\n')
# print(f'job counts: {df["job"].value_counts()}\n')

""" PREPROCESSING (LABEL ENCODING) """

# df = df.drop(columns=['month','day','loan','marital'])

label_encoder = LabelEncoder()
categoricals = df.select_dtypes(include="object").columns
encoded = df.copy()

for col in categoricals:
    encoded[col] = label_encoder.fit_transform(df[col]) 

x = encoded.drop('y',axis=1)
y = encoded['y']

""" SMOTE OVERSAMPLING """

oversample = SMOTE()
x,y = oversample.fit_resample(x,y)

""" OPTIMIZATION """

scaler = StandardScaler()
x = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30)

""" LOGISTIC REGRESSION """

logReg = LogisticRegression()
logReg.fit(X_train, y_train)

coefficients = logReg.coef_[0]
# for feature, coef in zip(feature_names, coefficients):
#     print(f"{feature}: {coef}")

""" CONFUSION MATRIX AND METRICS """

y_pred = logReg.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# print(f'Metrics')
print(f'Logistic Regression Score: {logReg.score(X_test,y_test)}')
print(f'Logistic Regression Macro F-1 Score: {macro_f1}')
print(f'Logistic Regression Precision: {precision}')
print(f'Logistic Regression Recall: {recall}')

result = permutation_importance(logReg, X_test, y_test, n_repeats=10, random_state=30)

# sort features by importance and select the top 10
sorted_idx = result.importances_mean.argsort()[-10:][::-1]  # Top 10 in descending order
top_features = encoded.columns[sorted_idx]
top_importances = result.importances_mean[sorted_idx]

# top 10 important features
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances, color="skyblue")
plt.xlabel("Permutation Importance")
plt.title("Top 10 Important Features in Logistic Regression")
plt.gca().invert_yaxis()  # Invert y-axis to have the most important on top
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

print(f'Confusion Matrix Score: {(cm[0][0]+cm[1][1])/len(X_test)}')

disp.plot()
plt.show()