import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('KOSPI_FRAME_ver3.csv', encoding='CP949')
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
        # frame = frame.dropna()
        frame = frame.drop([0, 1, 2], 0)
        # print(len(features))

        # feature = ['LREMTTTTUSM156S', 'FEDFUNDS', '한국실업률']  # EXTRA
        # feature = ['LREMTTTTUSM156S', 'LRUNTTTTKRM156S', 'IR3TCD01KRM156N']  # RANDOM
        features = ['DATE', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Volume', 'FEDFUNDS', 'USSLIND']  # XGB
        feature = ['Volume', 'FEDFUNDS', 'USSLIND']
        # print(feature)
        # frame[feature] = frame[feature].apply(lambda x: x.astype(float))
        Dependent = 'HM4UP'
        x = frame[features]
        y = frame[[Dependent]]
        # print(frame.shape)
        X_train = frame[features].iloc[:108, :]
        y_train = frame[Dependent].iloc[:108]
        X_test = frame[features].iloc[108:, :]
        y_test = frame[Dependent].iloc[108:]

        sc = StandardScaler()
        sc.fit(X_train[feature])
        X_train_std = sc.transform(X_train[feature])
        X_test_std = sc.transform(X_test[feature])
        # ml = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=0)
        # ml = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=7, gamma=0)
        ml.fit(X_train_std, y_train)
        y_pred = ml.predict(X_test_std)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print('accuracy : %.3f' %accuracy)
        print('precision : %.3f' %precision)
        print('recall : %.3f' %recall)

        X_test['HM4UP'] = y_test
        X_test['pred'] = y_pred
        X_test.loc[111, 'accuracy'] = accuracy
        X_test.loc[111, 'precision'] = precision
        X_test.loc[111, 'recall'] = recall
        X_test.to_csv("XGB_Kospi_HM4UP_result.csv", encoding='cp949')
