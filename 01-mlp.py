import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

df = pd.read_csv('./datasets/marketing_data.csv',sep=';')

""" DATA OBSERVATION """

# print(f'numeric: {df.select_dtypes(include="number").columns}')
# print(f'categorical: {df.select_dtypes(include="object").columns}')

""" PREPROCESSING (LABEL ENCODING) """
label_encoder = LabelEncoder()
categoricals = df.select_dtypes(include="object").columns
encoded = df.copy()

for col in categoricals:
    encoded[col] = label_encoder.fit_transform(df[col]) 

x = encoded.drop('y',axis=1)
y = encoded['y']

""" SMOTE OVERSAMPLING """

# print(f'Pre-SMOTE x-shape: {x.shape}')
# print(f'Pre-SMOTE y-shape: {y.shape}')

oversample = SMOTE()
x,y = oversample.fit_resample(x,y)

# print(f'Post-SMOTE x-shape: {x.shape}')
# print(f'Post-SMOTE y-shape: {y.shape}')

""" OPTIMIZATION """
scaler = StandardScaler()
x = scaler.fit_transform(x)

""" MULTILAYER PERCEPTRON """

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30)

mlpClassifier = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlpClassifier.fit(X_train, y_train)

""" CONFUSION MATRIX AND METRICS """

y_pred = mlpClassifier.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f'MLP Score: {mlpClassifier.score(X_test,y_test)}')
print(f'MLP Macro F1-Score: {macro_f1}')
print(f'MLP Macro Precision: {precision}')
print(f'MLP Macro Recall: {recall}')

result = permutation_importance(mlpClassifier, X_test, y_test, n_repeats=10, random_state=30)

# sort features by importance and select the top 10
sorted_idx = result.importances_mean.argsort()[-10:][::-1]  # Top 10 in descending order
top_features = encoded.columns[sorted_idx]
top_importances = result.importances_mean[sorted_idx]

# top 10 important features
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances, color="skyblue")
plt.xlabel("Permutation Importance")
plt.title("Top 10 Important Features in MLP Classifier")
plt.gca().invert_yaxis()  # Invert y-axis to have the most important on top
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(f'Confusion Matrix Score: {(cm[0][0]+cm[1][1])/len(X_test)}')
