from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn import datasets
import pydotplus
from linear_algebra import dot
import random, math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn import metrics
import statsmodels.api as sm
import glob
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def logistic(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))

if __name__ == '__main__':

    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\decision_trees\종속변수\1차 지표'
    allFiles = glob.glob(path + '/*.csv')
    frame = pd.DataFrame()
    list_ = []
    cnt = 0

    for file_ in allFiles:
        print("read " + file_)
        df = pd.read_csv(file_, encoding='CP949')
        if cnt == 0:
            frame = df
            cnt = cnt + 1
        else:
            frame = pd.merge(frame, df, on='DATE')

    dow = pd.read_csv('HM4UP.csv', encoding='CP949')

    frame = pd.merge(frame, dow, on='DATE')
    frame = frame.dropna()
    frame = frame.set_index('DATE')
    frame = frame.sort_values('DATE')

    feature = ["미국 내구재 주문", "미국 소비자 물가 상승률", "미국 소비율"]
    feature_cols = ["US Durable Goods Order", "US consumer price inflation", "US consumption rate"]
    Dependent = 'HM4UP'
    dfx = frame[feature]
    dfy = frame[[Dependent]]

    random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(frame[feature], frame[Dependent], test_size=0.2)
    # y_test.to_csv('y_test.csv', encoding='CP949')

    # X_test = frame[feature].iloc[292:, :]
    # X_train = frame[feature].iloc[:292, :]
    # y_train = frame[Dependent].iloc[:292]
    # y_test = frame[Dependent].iloc[292:]

    print()
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    random.seed(0)
    true_positives = false_positives = true_negatives = false_negatives = 0
    dff = pd.DataFrame()

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    x2 = sm.add_constant(dfx)
    model = sm.OLS(dfy, x2)
    result = model.fit()
    print(result.summary())

    y_pred = log_reg.predict(X_test)
    print(y_pred)
    print(y_test)

    print('정확도 :', metrics.accuracy_score(y_test, y_pred))
    i = 0
    for x_i, y_i in zip(y_pred, y_test):
        if y_i == 1 and x_i == 1:  # TP: paid and we predict paid
            true_positives += 1
            dff.loc[i, 'TP'] = 1
        elif y_i == 1:  # FN: paid and we predict unpaid
            false_negatives += 1
            dff.loc[i, 'FN'] = 1
        elif x_i >= 0.6:  # FP: unpaid and we predict paid
            false_positives += 1
            dff.loc[i, 'FP'] = 1
        else:  # TN: unpaid and we predict unpaid
            true_negatives += 1
            dff.loc[i, 'TN'] = 1
        i = i + 1

    dff.to_csv("SKL.csv", encoding="cp949")