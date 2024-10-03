import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

df = pd.read_csv('./datasets/marketing_data.csv',sep=';')

""" NO SAMPLING MODIFICATIONS """

# df_yes = df[df["y"] == "yes"]
# df_no = df[df["y"] == "no"]

# df_no_downsampled = resample(
#     df_no,
#     replace=False,
#     n_samples=len(df_yes),
#     random_state=42
# )

# df_balanced = pd.concat([df_yes, df_no_downsampled])
# df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

""" OVER SAMPLING WITH SMOTE """

oversample = SMOTE()


""" DOWNSAMPLING"""

""" SMOTE """

""" SMOTE + ENN """