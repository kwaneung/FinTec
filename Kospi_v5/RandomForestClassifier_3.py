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
from xgboost import XGBClassifier

if __name__ == '__main__':
    frame = pd.read_csv('KOSPI_FRAME3.csv', encoding='CP949')
    features = list(frame.keys())

    features.remove('DATE')
    features.remove('Open')
    features.remove('High')
    features.remove('Low')
    features.remove('Close')
    features.remove('Adj Close')
    features.remove('HM4UP')
    features.remove('LM4DN')
    for i in range(len(features)):
        if features[i] == 'Volume':  # Volume은 shift 되면 안되지만 features에는 들어있어야함.
            continue
        frame[features[i]] = frame[features[i]].shift(3)
    # frame['경상수지'] = frame['경상수지'].shift(+3)
    # frame['상품수지'] = frame['상품수지'].shift(+3)
    # frame['BAA'] = frame['BAA'].shift(+3)
    # frame['UNRATE'] = frame['UNRATE'].shift(+3)
    # frame['NEWORDER'] = frame['NEWORDER'].shift(+3)
    # frame['CSUSHPINSA'] = frame['CSUSHPINSA'].shift(+3)
    # frame['LREMTTTTUSM156S'] = frame['LREMTTTTUSM156S'].shift(+3)
    # frame['DGORDER'] = frame['DGORDER'].shift(+3)
    # frame['TWEXBMTH'] = frame['TWEXBMTH'].shift(+3)
    # frame['UNRATENSA'] = frame['UNRATENSA'].shift(+3)
    # frame['TCU'] = frame['TCU'].shift(+3)
    # frame['INDPRO'] = frame['INDPRO'].shift(+3)
    # frame['PPIACO'] = frame['PPIACO'].shift(+3)
    # frame['RMFSL'] = frame['RMFSL'].shift(+3)
    # frame['CPIAUCSL'] = frame['CPIAUCSL'].shift(+3)
    # frame['HOUST'] = frame['HOUST'].shift(+3)
    # frame['HSN1F'] = frame['HSN1F'].shift(+3)
    # frame['FEDFUNDS'] = frame['FEDFUNDS'].shift(+3)
    # frame['USSLIND'] = frame['USSLIND'].shift(+3)
    # frame['TOTALSA'] = frame['TOTALSA'].shift(+3)
    # frame['UMCSENT'] = frame['UMCSENT'].shift(+3)
    # frame['서울아파트매매가격지수'] = frame['서울아파트매매가격지수'].shift(+3)
    # frame['AMBNS'] = frame['AMBNS'].shift(+3)
    # frame['자가주거비포함지수'] = frame['자가주거비포함지수'].shift(+3)
    # frame['자가주거비'] = frame['자가주거비'].shift(+3)
    # frame['수출물가지수'] = frame['수출물가지수'].shift(+3)
    # frame['XTEXVA01CNM667S'] = frame['XTEXVA01CNM667S'].shift(+3)
    # frame['정책금리'] = frame['정책금리'].shift(+3)
    # frame['통화량'] = frame['통화량'].shift(+3)
    # frame['XTIMVA01KRM667S'] = frame['XTIMVA01KRM667S'].shift(+3)
    # frame['KORPROINDMISMEI'] = frame['KORPROINDMISMEI'].shift(+3)
    # frame['KORCPIALLMINMEI'] = frame['KORCPIALLMINMEI'].shift(+3)
    # frame['한국실업률'] = frame['한국실업률'].shift(+3)
    # frame['IR3TCD01KRM156N'] = frame['IR3TCD01KRM156N'].shift(+3)
    # frame['환율평균'] = frame['환율평균'].shift(+3)
    frame = frame.drop([0, 1, 2], 0)

    y_result = pd.DataFrame()
    # for t in [5, 6, 7, 8]:
    #     for p in [0, 1]:
    #         print()
    #         print(str(t) + ' 깊이')
    #         print(str(p) + ' 감마')
    #         print()
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
                X_train = frame[features].iloc[:84, :]
                y_train = frame[Dependent].iloc[:84]
                X_test = frame[features].iloc[84:, :]
                y_test = frame[Dependent].iloc[84:]

                # sc = StandardScaler()
                # sc.fit(X_train)
                # X_train_std = sc.transform(X_train)
                # X_test_std = sc.transform(X_test)

                # ml = XGBClassifier(n_estimators=100, min_child_weight=1, n_jobs=-1, max_depth=t, gamma=p)
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

                y_result.loc[cnt, 'feature'] = str(feature)
                y_result.loc[cnt, 'accuracy'] = accuracy
                y_result.loc[cnt, 'precision'] = precision
                y_result.loc[cnt, 'recall'] = recall
                cnt = cnt + 1
                print("%.2f %%" % (cnt * 100 / 11480))

    y_result.to_csv("_XGB_HM4UP_Kospi_result.csv", encoding='CP949')

    print("time :", time.time() - start)
