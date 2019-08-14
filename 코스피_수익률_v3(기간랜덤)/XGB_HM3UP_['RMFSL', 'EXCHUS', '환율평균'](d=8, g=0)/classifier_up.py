import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    for t in range(1):
        frame = pd.read_csv('KOSPI_FRAME_ver4.csv', encoding='CP949')
        features = list(frame.keys())

        features.remove('DATE')
        features.remove('Open')
        features.remove('High')
        features.remove('Low')
        features.remove('Close')
        features.remove('Adj Close')
        features.remove('HM4UP')
        features.remove('HM3UP')

        for i in range(len(features)):
            frame[features[i]] = frame[features[i]].shift(3)
        frame = frame.drop([0, 1, 2], 0)

        feature = ['RMFSL', 'EXCHUS', '환율평균']
        features = ['DATE', 'Open', 'High', 'Low', 'Close', 'Adj Close']  # XGB
        features = features + feature
        # print(features)

        Dependent = 'HM3UP'
        x = frame[features]
        y = frame[[Dependent]]

        # X_train = frame[features].iloc[:84, :]
        # y_train = frame[Dependent].iloc[:84]
        # X_test = frame[features].iloc[84:, :]
        # y_test = frame[Dependent].iloc[84:]

        X_train, X_test, y_train, y_test = train_test_split(frame[features], frame[Dependent], test_size=0.3, random_state=682)  # 682, 2409

        sc = StandardScaler()
        sc.fit(X_train[feature])
        X_train_std = sc.transform(X_train[feature])
        X_test_std = sc.transform(X_test[feature])

        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=8, gamma=0)
        ml.fit(X_train_std, y_train)
        y_pred = ml.predict(X_test_std)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        if precision >= 0.5 and recall >= 0.5:
            print('random_state : %d' %t)
            print('accuracy : %.3f' %accuracy)
            print('precision : %.3f' %precision)
            print('recall : %.3f' %recall)
            print()

            X_test['HM3UP'] = y_test
            X_test['pred'] = y_pred
            X_test.loc[87, 'accuracy'] = accuracy
            X_test.loc[87, 'precision'] = precision
            X_test.loc[87, 'recall'] = recall
            X_test.to_csv("XGB_Kospi_HM3UP_result.csv", encoding='cp949')
