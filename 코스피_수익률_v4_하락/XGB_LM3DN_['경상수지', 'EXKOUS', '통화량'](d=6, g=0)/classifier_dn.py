import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('KOSPI_FRAME_ver3(12Y).csv', encoding='CP949')
        features = list(frame.keys())

        features.remove('DATE')
        features.remove('Open')
        features.remove('High')
        features.remove('Low')
        features.remove('Close')
        features.remove('Adj Close')
        features.remove('LM4DN')
        features.remove('LM3DN')

        for i in range(len(features)):
            frame[features[i]] = frame[features[i]].shift(3)
        frame = frame.drop([0, 1, 2], 0)

        feature = ['경상수지', 'EXKOUS', '통화량']
        features = ['DATE', 'Open', 'High', 'Low', 'Close', 'Adj Close']  # XGB
        features = features + feature
        print(features)

        Dependent = 'LM3DN'
        x = frame[features]
        y = frame[[Dependent]]

        # short
        X_train = frame[features].iloc[:108, :]
        y_train = frame[Dependent].iloc[:108]
        X_test = frame[features].iloc[108:, :]
        y_test = frame[Dependent].iloc[108:]

        sc = StandardScaler()
        sc.fit(X_train[feature])
        X_train_std = sc.transform(X_train[feature])
        X_test_std = sc.transform(X_test[feature])

        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=6, gamma=0)
        ml.fit(X_train_std, y_train)
        y_pred = ml.predict(X_test_std)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print('accuracy : %.3f' % accuracy)
        print('precision : %.3f' % precision)
        print('recall : %.3f' % recall)

        X_test['LM3DN'] = y_test
        X_test['LM3DN_pred'] = y_pred
        X_test.to_csv("XGB_Kospi_LM3DN_result.csv", encoding='cp949')