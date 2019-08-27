import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from itertools import combinations

# 학습 기간을 슬라이딩 윈도우 방식으로 총기간 5년으로 최근까지 6개월씩 시프트 하면서 가보자.
if __name__ == '__main__':
    frame = pd.read_csv('KOSPI_FRAME_ver3(12Y).csv', encoding='CP949')
    features = list(frame.keys())

    features.remove('DATE')
    features.remove('Open')
    features.remove('High')
    features.remove('Low')
    features.remove('Close')
    features.remove('Adj Close')
    features.remove('LM3DN')
    features.remove('LM4DN')
    features.remove('HM3UP')
    features.remove('HM4UP')

    for i in range(len(features)):
        frame[features[i]] = frame[features[i]].shift(3)
    # print(i)
    frame = frame.drop([0, 1, 2], 0)

    y_result = pd.DataFrame()  # 각 사이클의 결과종합

    windowSize = 60  # 윈도우 사이즈 5년 = 60개월
    slideSize = 6  # 6개월씩 슬라이드
    training_ratio = 0.7  # 트레이닝 비율
    frameSize = len(frame)  # 총 12년이니까 144개월

    t = 4  # 깊이
    cnt = 0

    feature = ['UNRATE', '정책금리', 'LRUNTTTTKRM156S']

    avgAccuracy = 0
    avgPrecision = 0
    avgRecall = 0
    avgReturns = 0

    f = open("result.txt", 'w')
    f.write("Accuracy" + " " + "Precision" + " " + "Recall" + " " + "Returns" + "\n")
    for sSize in range(int((frameSize - windowSize) / slideSize) + 1):  # 전체 기간에서 윈도우 크기만큼 뺀 기간을 6개월로 나누게되면 윈도우의 움직일 횟수가 나옴. + 첫번째 한번 돌릴거 추가
        # 0 <= sSize <= 14 으로 총 15번
        # sSize 반복문의 한사이클이 돌면 precision과 recall과 수익률의 평균을 내야함.
        # print(sSize)
        # sSize * 6이 시작 위치
        resizeFrame = frame[sSize * 6:windowSize + sSize * 6]  # 윈도우 사이즈와 간격만큼 원본 frame에서 뽑아 resizeFrame에 넣는다.
        # print(len(resizeFrame))
        # feature = [features[i], features[j], features[k]]

        # print(feature)
        # print(len(features))
        Dependent = 'HM3UP'
        x = resizeFrame[feature]
        y = resizeFrame[[Dependent]]

        X_train = resizeFrame[feature].iloc[:int(windowSize * training_ratio), :]  # 108대신에 총 길이 * training_ratio 한뒤 int 변환
        y_train = resizeFrame[Dependent].iloc[:int(windowSize * training_ratio)]
        X_test = resizeFrame[feature].iloc[int(windowSize * training_ratio):, :]
        y_test = resizeFrame[Dependent].iloc[int(windowSize * training_ratio):]

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        ml = XGBClassifier(n_estimators=100, min_child_weight=1, max_depth=t)

        ml.fit(X_train_std, y_train)
        y_pred = ml.predict(X_test_std)

        y_pred = pd.DataFrame(data={Dependent + '_pred': y_pred})
        y_test = pd.DataFrame(data={Dependent: y_test}).reset_index().drop('index', axis=1)
        date = resizeFrame['DATE'].iloc[int(windowSize * training_ratio):].reset_index().drop('index', axis=1)
        kValues = resizeFrame[['Open', 'High', 'Low', 'Close', 'Adj Close']].iloc[int(windowSize * training_ratio):, :]

        result = pd.concat([date, kValues.reset_index(), X_test.reset_index(), y_test, y_pred], axis=1).drop('index', axis=1)  # 한개의 예측 결과물.

        # 파라미터1 : 조건, 2 : 참인경우, 3: 거짓인 경우
        result['revenue'] = np.where(result.HM3UP_pred, np.where(result.High >= result.Open * 1.04, result.Open * 0.04, result.Close - result.Open), 0)

        Accuracy = accuracy_score(y_test, y_pred)
        Precision = precision_score(y_test, y_pred)
        Recall = recall_score(y_test, y_pred)
        Returns = (result['revenue'].sum() / result.loc[0, 'Close'])
        f.write(str(Accuracy) + " " + str(Precision) + " " + str(Recall) + " " + str(Returns) + "\n")


