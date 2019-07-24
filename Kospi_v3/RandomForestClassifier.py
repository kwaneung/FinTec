import time
start = time.time()  # 시작 시간 저장
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import random
import xgboost as xgb

# 볼륨을 예측 : 범주형 데이터
if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('NASDAQ_FRAME.csv', encoding='CP949')
    features = list(frame.keys())
    feature = ['Volume', '서울아파트매매가격지수', 'EXCHUS', 'T10Y2YM']
    # print(feature)
    frame[feature] = frame[feature].apply(lambda x: x.astype(float))
    Dependent = 'LM4D'
    x = frame[feature]
    y = frame[[Dependent]]

    X_train = frame[feature].iloc[:130, :]
    y_train = frame[Dependent].iloc[:130]
    X_test = frame[feature].iloc[130:, :]
    y_test = frame[Dependent].iloc[130:]

    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)
    ml = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    # ml = xgb.XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=6, gamma=0.01)
    ml.fit(X_train, y_train)
    y_pred = ml.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('accuracy : %.3f' %accuracy)
    print('precision : %.3f' %precision)
    print('recall : %.3f' %recall)

    print(list(y_test))
    print(ml.predict(X_test))
    tt = pd.DataFrame(ml.predict(X_test))
    test = pd.concat([frame['DATE'], X_test, y_test, tt])
    test.to_csv("result_3_XGB.csv", encoding='cp949')

    print("time :", time.time() - start)
