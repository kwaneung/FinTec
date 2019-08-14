import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if __name__ == '__main__':
    frame = pd.read_csv('KOSPI_FRAME_ver3(12Y).csv', encoding='CP949')  # long
    # frame = pd.read_csv('KOSPI_FRAME_ver4.csv', encoding='CP949')  # short
    features = list(frame.keys())

    features.remove('DATE')
    features.remove('Open')
    features.remove('High')
    features.remove('Low')
    features.remove('Close')
    features.remove('Adj Close')
    features.remove('LM3DN')
    features.remove('LM4DN')

    for i in range(len(features)):
        frame[features[i]] = frame[features[i]].shift(3)
    # print(i)
    frame = frame.drop([0, 1, 2], 0)

    y_result = pd.DataFrame()
    for t in [6, 7, 8, 9]:
        for p in [0, 1]:
            print()
            print(str(t) + ' 깊이')
            print(str(p) + ' 감마')
            print()
            cnt = 0
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    for k in range(j+1, len(features)):
                        feature = [features[i], features[j], features[k]]

                        # frame[feature] = frame[feature].apply(lambda x: x.astype(float))
                        Dependent = 'LM4DN'
                        x = frame[feature]
                        y = frame[[Dependent]]

                        X_train = frame[feature].iloc[:108, :]
                        y_train = frame[Dependent].iloc[:108]
                        X_test = frame[feature].iloc[108:, :]
                        y_test = frame[Dependent].iloc[108:]

                        sc = StandardScaler()
                        sc.fit(X_train)
                        X_train_std = sc.transform(X_train)
                        X_test_std = sc.transform(X_test)

                        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=t, gamma=p)

                        ml.fit(X_train_std, y_train)
                        y_pred = ml.predict(X_test_std)

                        accuracy = accuracy_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)

                        y_result.loc[cnt, 'feature'] = str(feature)
                        y_result.loc[cnt, 'accuracy'] = accuracy
                        y_result.loc[cnt, 'precision'] = precision
                        y_result.loc[cnt, 'recall'] = recall
                        cnt = cnt + 1
                        print("%.2f %%" % (cnt * 100 / 11480))

            y_result.to_csv(str(t) + "_" + str(p) + "_" + "XGB_STD_LM4DN_LM3DN_result.csv", encoding='CP949')
