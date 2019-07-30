import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

if __name__ == '__main__':
    for i in range(1):
        frame = pd.read_csv('KOSPI_FRAME.csv', encoding='CP949')

        frame['경상수지'] = frame['경상수지'].shift(+2)
        frame['상품수지'] = frame['상품수지'].shift(+2)
        frame['BAA'] = frame['BAA'].shift(+1)
        frame['UNRATE'] = frame['UNRATE'].shift(+1)
        frame['NEWORDER'] = frame['NEWORDER'].shift(+2)
        frame['CSUSHPINSA'] = frame['CSUSHPINSA'].shift(+2)
        frame['LREMTTTTUSM156S'] = frame['LREMTTTTUSM156S'].shift(+1)
        frame['DGORDER'] = frame['DGORDER'].shift(+2)
        frame['TWEXBMTH'] = frame['TWEXBMTH'].shift(+1)
        frame['UNRATENSA'] = frame['UNRATENSA'].shift(+1)
        frame['TCU'] = frame['TCU'].shift(+1)
        frame['INDPRO'] = frame['INDPRO'].shift(+1)
        frame['PPIACO'] = frame['PPIACO'].shift(+1)
        frame['RMFSL'] = frame['RMFSL'].shift(+1)
        frame['CPIAUCSL'] = frame['CPIAUCSL'].shift(+1)
        frame['HOUST'] = frame['HOUST'].shift(+2)
        frame['HSN1F'] = frame['HSN1F'].shift(+1)
        frame['FEDFUNDS'] = frame['FEDFUNDS'].shift(+1)
        frame['USSLIND'] = frame['USSLIND'].shift(+2)
        frame['TOTALSA'] = frame['TOTALSA'].shift(+1)
        frame['UMCSENT'] = frame['UMCSENT'].shift(+1)
        frame['서울아파트매매가격지수'] = frame['서울아파트매매가격지수'].shift(+1)
        frame['AMBNS'] = frame['AMBNS'].shift(+1)
        frame['자가주거비포함지수'] = frame['자가주거비포함지수'].shift(+1)
        frame['자가주거비'] = frame['자가주거비'].shift(+1)
        frame['수출물가지수'] = frame['수출물가지수'].shift(+1)
        frame['XTEXVA01CNM667S'] = frame['XTEXVA01CNM667S'].shift(+2)
        frame['정책금리'] = frame['정책금리'].shift(+3)
        frame['통화량'] = frame['통화량'].shift(+3)
        frame['XTIMVA01KRM667S'] = frame['XTIMVA01KRM667S'].shift(+2)
        frame['KORPROINDMISMEI'] = frame['KORPROINDMISMEI'].shift(+2)
        frame['KORCPIALLMINMEI'] = frame['KORCPIALLMINMEI'].shift(+1)
        frame['한국실업률'] = frame['한국실업률'].shift(+1)
        frame['IR3TCD01KRM156N'] = frame['IR3TCD01KRM156N'].shift(+1)
        frame['환율평균'] = frame['환율평균'].shift(+1)
        # frame = frame.dropna()
        frame = frame.drop([0, 1, 2], 0)
        # print(len(features))

        # feature = ['LREMTTTTUSM156S', 'FEDFUNDS', '한국실업률']  # EXTRA
        # feature = ['LREMTTTTUSM156S', 'LRUNTTTTKRM156S', 'IR3TCD01KRM156N']  # RANDOM
        features = ['DATE', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'ISM_PMI_Actual', 'GACDFSA066MSFRBPHI', '한국실업률']  # XGB
        feature = ['ISM_PMI_Actual', 'GACDFSA066MSFRBPHI', '한국실업률']
        # print(feature)
        # frame[feature] = frame[feature].apply(lambda x: x.astype(float))
        Dependent = 'HM4UP'
        x = frame[features]
        y = frame[[Dependent]]
        # print(frame.shape)
        X_train = frame[features].iloc[:106, :]
        y_train = frame[Dependent].iloc[:106]
        X_test = frame[features].iloc[106:, :]
        y_test = frame[Dependent].iloc[106:]

        # sc = StandardScaler()
        # sc.fit(X_train)
        # X_train_std = sc.transform(X_train)
        # X_test_std = sc.transform(X_test)
        # ml = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=0)
        # ml = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=6, gamma=0)
        ml.fit(X_train[feature], y_train)
        y_pred = ml.predict(X_test[feature])

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print('accuracy : %.3f' %accuracy)
        print('precision : %.3f' %precision)
        print('recall : %.3f' %recall)

        X_test['HM4UP'] = y_test
        X_test['pred'] = y_pred
        X_test.to_csv("XGB_Kospi_HM4UP_result.csv", encoding='cp949')
