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
if __name__ == '__main__':
    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\Kospi\종속변수\새 폴더'
    allFiles = glob.glob(path + '/*.csv')
    frame = pd.DataFrame()
    list_ = []
    cnt = 0

    for file_ in allFiles:
        print("read " + file_)
        df = pd.read_csv(file_, encoding='CP949')
        if cnt == 0:
            frame = df
            cnt = cnt + 1
        else:
            frame = pd.merge(frame, df, on='DATE')

    HM4UP = pd.read_csv('HM4UP.csv', encoding='CP949')
    frame = pd.merge(frame, HM4UP, on='DATE')

    frame = frame.dropna()
    frame = frame.set_index('DATE')
    frame.to_csv("KOSPI_FRAME.csv",  encoding='CP949')
    # frame = frame[["HM4UP","미국 신규주택착공", "원달러 환율", "미국 신규 실업수당 청구건수", "미국의 소매판매지수", "미국 내구재 주문", "미국 소비자 물가 상승률", "미국 소비자 신뢰지수"]]

    corr = frame.corr(method='pearson')
    corr.to_csv('corr.csv', encoding='CP949')

    feature = ["미국 신규주택착공", "원달러 환율", "미국 신규 실업수당 청구건수", "미국의 소매판매지수"]
    frame[feature] = frame[feature].apply(lambda x: x.astype(float))
    Dependent = 'HM4UP'
    x = frame[feature]
    y = frame[[Dependent]]

    X_test = frame[feature].iloc[100:, :]
    X_train = frame[feature].iloc[:100, :]
    y_train = frame[Dependent].iloc[:100]
    y_test = frame[Dependent].iloc[100:]

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    ml = ExtraTreesClassifier(criterion='entropy', n_estimators=500, n_jobs=-1, random_state=0)
    ml.fit(X_train_std, y_train)
    y_pred = ml.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('accuracy : %.3f' %accuracy)
    print('precision : %.3f' %precision)
    print('recall : %.3f' %recall)

    print(y_pred)