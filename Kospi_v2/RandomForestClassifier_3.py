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

if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('KOSPI_FRAME.csv', encoding='CP949')
        features = list(frame.keys())

        y_prec = pd.DataFrame()
        y_reca = pd.DataFrame()
        features.remove('DATE')
        features.remove('Adj Close')
        features.remove('HM4UP')
        features.remove('LM4DN')
        # print(len(features))
        cnt = 0

        for i in range(len(features)):
            for j in range(i+1, len(features)):
                for k in range(j+1, len(features)):
                    feature = [features[i], features[j], features[k]]
                    # print(feature)
                    frame[feature] = frame[feature].apply(lambda x: x.astype(float))
                    Dependent = 'HM4UP'
                    x = frame[feature]
                    y = frame[[Dependent]]
                    # print(frame.shape)
                    X_train = frame[feature].iloc[:108, :]
                    y_train = frame[Dependent].iloc[:108]
                    X_test = frame[feature].iloc[108:, :]
                    y_test = frame[Dependent].iloc[108:]

                    # sc = StandardScaler()
                    # sc.fit(X_train)
                    # X_train_std = sc.transform(X_train)
                    # X_test_std = sc.transform(X_test)
                    ml = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
                    ml.fit(X_train, y_train)
                    y_pred = ml.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)

                    # print('accuracy : %.3f' %accuracy)
                    # print('precision : %.3f' %precision)
                    # print('recall : %.3f' %recall)
                    # print(y_pred)

                    y_prec.loc[cnt, 'feature'] = str(feature)
                    y_prec.loc[cnt, 'precision'] = precision
                    y_reca.loc[cnt, 'feature'] = str(feature)
                    y_reca.loc[cnt, 'recall'] = recall
                    cnt = cnt + 1
                    print("%.2f %%" % (cnt * 100 / 13244))

    y_prec.to_csv("y_precision_LM4D_result.csv", encoding='CP949')
    y_reca.to_csv("y_recall_LM4D_result.csv", encoding='CP949')

    print("time :", time.time() - start)
