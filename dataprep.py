import pandas as pd
import torch

def fix_malefemale(df):
    df["gender"] = df.gender.replace('f', 'female')
    df["gender"] = df.gender.replace('m', 'female')
    return df


def onehot_encode(df):
    onehot_rules = {
        'gender': ['male', 'female'],
        'cholesterol': ['medium', 'low', 'high'],
        'gluc': ['medium', 'low', 'high'],
    }

    for col, vals in onehot_rules.items():
        for val in vals:
            newcol_name = f"{col}_{val}"
            df[newcol_name] = (df[col] == val).astype('int')
        df = df.drop(col, axis=1)

    return df


def split_x_y(df, label='diabetes'):
    y = df.diabetes
    X = df.drop(label, axis=1)

    X=torch.Tensor(X.values.astype('float'))
    y=torch.Tensor(y.values.astype('int'))
    return X, y


def split_pressure(df):
    newcols = df.pressure.str.split('/', expand=True)
    newcols.columns = ['pressure_0', 'pressure_1']
    df = df.join(newcols)
    df = df.drop('pressure', axis=1)
    return df


def read_data():

    df_train_info = pd.read_csv('./diabetes_v2/diabetes_train_info.csv', index_col=0)
    df_train_analysis = pd.read_csv('./diabetes_v2/diabetes_train_analysis.csv', index_col=0)
    df_train = df_train_info.join(df_train_analysis)

    df_test_info = pd.read_csv('./diabetes_v2/diabetes_test_info.csv', index_col=0)
    df_test_analysis = pd.read_csv('./diabetes_v2/diabetes_test_analysis.csv', index_col=0)
    df_test = df_test_info.join(df_test_analysis)

    df_train=(df_train.pipe(fix_malefemale)
              .pipe(onehot_encode)
              .pipe(split_pressure)
              ).dropna()
    X_train, y_train = split_x_y(df_train)


    df_test=(df_test.pipe(fix_malefemale)
             .pipe(onehot_encode)
             .pipe(split_pressure)
             ).dropna()
    X_test, y_test = split_x_y(df_test)

    X_train = torch.nn.functional.normalize(X_train, dim=0)
    X_test = torch.nn.functional.normalize(X_test, dim=0)



    X_train -= X_train.mean(dim=0)
    X_train/=X_train.abs().max(axis=0).values

    X_test -= X_test.mean(dim=0)
    X_test /= X_test.abs().max(axis=0).values


    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)