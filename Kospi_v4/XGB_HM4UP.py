import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('KOSPI_FRAME2.csv', encoding='CP949')

        frame['상품수지'] = frame['상품수지'].shift(+2)
        frame['LREMTTTTUSM156S'] = frame['LREMTTTTUSM156S'].shift(+1)
        frame['UNRATENSA'] = frame['UNRATENSA'].shift(+1)

        frame = frame.drop([0, 1, 2], 0)  # 시프트로인한 널값 제거

        features = ['DATE', 'Open', 'High', 'Low', 'Close', 'Adj Close', '상품수지', 'LREMTTTTUSM156S', 'UNRATENSA']
        feature = ['상품수지', 'LREMTTTTUSM156S', 'UNRATENSA']

        Dependent = 'HM4UP'
        x = frame[features]
        y = frame[[Dependent]]

        X_train = frame[features].iloc[:106, :]
        y_train = frame[Dependent].iloc[:106]
        X_test = frame[features].iloc[106:, :]
        y_test = frame[Dependent].iloc[106:]

        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=7, gamma=0)
        ml.fit(X_train[feature], y_train)
        y_pred = ml.predict(X_test[feature])

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print('accuracy : %.3f' % accuracy)
        print('precision : %.3f' % precision)
        print('recall : %.3f' % recall)

        X_test['HM4UP'] = y_test
        X_test['pred'] = y_pred
        X_test.to_csv("XGB_Kospi_HM4UP_result.csv", encoding='cp949')

