import time
start = time.time()
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import random
from xgboost import XGBClassifier

if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('NASDAQ_FRAME.csv', encoding='CP949')
        features = list(frame.keys())
        y_prec = pd.DataFrame()
        y_reca = pd.DataFrame()
        features.remove('DATE')
        features.remove('Open')
        features.remove('High')
        features.remove('Low')
        features.remove('Close')
        features.remove('Adj Close')
        features.remove('HM4UP')
        features.remove('LM4D')
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
                    X_train = frame[feature].iloc[:130, :]
                    y_train = frame[Dependent].iloc[:130]
                    X_test = frame[feature].iloc[130:, :]
                    y_test = frame[Dependent].iloc[130:]

                    # sc = StandardScaler()
                    # sc.fit(X_train)
                    # X_train_std = sc.transform(X_train)
                    # X_test_std = sc.transform(X_test)
                    # ml = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=0)
                    # ml = AdaBoostClassifier(n_estimators=100)
                    # ml = GradientBoostingClassifier(n_estimators=100, random_state=0)
                    # ml = xgb.XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=6, gamma=0)
                    ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=6, gamma=0)
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
                    print("%.2f %%" % (cnt * 100 / 9880))

    y_prec.to_csv("y_precision_HM4UP_result_3.csv", encoding='CP949')
    y_reca.to_csv("y_recall_HM4UP_result_3.csv", encoding='CP949')

    print("time :", time.time() - start)
