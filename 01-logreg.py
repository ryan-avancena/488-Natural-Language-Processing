import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.preprocessing import StandardScaler


df = pd.read_csv('./datasets/marketing_data.csv',sep=';')

""" DATA OBSERVATION """

print(f'numeric: {df.select_dtypes(include="number").columns}')
print(f'categorical: {df.select_dtypes(include="object").columns}')

""" PREPROCESSING """

# df = df.drop(columns=['month','day','loan','marital'])

label_encoder = LabelEncoder()
categoricals = df.select_dtypes(include="object").columns
encoded = df.copy()

for col in categoricals:
    encoded[col] = label_encoder.fit_transform(df[col]) 

""" LOGISTIC REGRESSION """

x = encoded.drop('y',axis=1)
y = encoded['y']


""" OPTIMIZATION """
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=30)

logReg = LogisticRegression()
logReg.fit(X_train_scaled, y_train)

# print(f'LogReg Coeffiecients: {logReg.coef_}')
# print(f'LogReg Bias: {logReg.intercept_}')
print(f'Logistic Regression Score: {logReg.score(X_test_scaled,y_test)}')


""" CONFUSION MATRIX """

y_pred = logReg.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

print(f'Confusion Matrix Score: {(cm[0][0]+cm[1][1])/len(X_test_scaled)}')

disp.plot()
plt.show()