import pandas
import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelBinarizer


def fix_malefemale(df):
    df["gender"] = df.gender.replace('f', 'female')
    df["gender"] = df.gender.replace('m', 'female')
    return df


def onehot_encode(df):
    onehot_rules={
        'gender': ['male', 'female'],
        'cholesterol': ['medium', 'low', 'high'],
        'gluc':   ['medium', 'low', 'high'],
    }

    for col, vals in onehot_rules.items():
        for val in vals:
            newcol_name = f"{col}_{val}"
            df[newcol_name] = (df[col]==val).astype('int')
        df = df.drop(col, axis=1)

    return df


def split_x_y(df, label='diabetes'):
    y=df.diabetes
    X = df.drop(label, axis=1)
    return X, y


def split_pressure(df):
    newcols = df.pressure.str.split('/', expand=True)
    newcols.columns = ['pressure_0', 'pressure_1']
    df = df.join(newcols)
    df= df.drop('pressure', axis=1)
    return df



df_train_info = pd.read_csv('./diabetes_v2/diabetes_train_info.csv', index_col=0)
df_train_analysis = pd.read_csv('./diabetes_v2/diabetes_train_analysis.csv', index_col=0)
df_train = df_train_info.join(df_train_analysis)


df_test_info = pd.read_csv('./diabetes_v2/diabetes_test_info.csv', index_col=0)
df_test_analysis = pd.read_csv('./diabetes_v2/diabetes_test_analysis.csv', index_col=0)
df_test = df_test_info.join(df_test_analysis)


df_train = fix_malefemale(df_train)
df_train = onehot_encode(df_train)
df_train = split_pressure(df_train)
df_train = df_train.dropna()
X_train, y_train = split_x_y(df_train)

df_test = fix_malefemale(df_test)
df_test = onehot_encode(df_test)
df_test = split_pressure(df_test)
df_test = df_test.dropna()
X_test, y_test = split_x_y(df_test)

"""
Index(['age', 'height', 'weight', 'gender', 'cholesterol', 'gluc', 'smoke',
       'alco', 'active', 'pressure', 'diabetes'],
       
# do math in pressure column

genders: {'male', 'f', 'female', 'm'}
cholesterol: {'medium', 'low', 'high'}
gluc:   {'medium', 'low', 'high'}
    
y = LabelBinarizer().fit_transform(df.Countries)
       
"""

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# Y = np.array([1, 1, 2, 2])
# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
SGDClassifier(max_iter=1000, tol=1e-3))
opt = clf.fit(X_train, y_train)
# Pipeline(steps=[('standardscaler', StandardScaler()),
#                 ('sgdclassifier', SGDClassifier())])
# >>> print(clf.predict([[-0.8, -1]]))

predictions = clf.predict(X_test)

from sklearn.metrics import precision_recall_curve, confusion_matrix
precision, recall, thresholds = precision_recall_curve(y_test, predictions)



cm=confusion_matrix(y_test, predictions)



#                     0/1
categorical_columns=['gender', 'cholesterol', 'gluc']





