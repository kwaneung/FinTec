import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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

    y_result = pd.DataFrame()

    windowSize = 60  # 5년 = 60개월
    slideSize = 6  # 6개월씩 슬라이드
    training_ratio = 0.7  # 0.6으로도 해보자.
    frameSize = len(frame)  # 총 12년이니까 144개월

    for sSize in range(int((frameSize-windowSize)/slideSize) + 1):  # 전체 기간에서 윈도우 크기만큼 뺀 기간을 6개월로 나누게되면 윈도우의 움직일 횟수가 나옴. + 첫번째 한번 돌릴거 추가
        # 0 <= sSize <= 14 으로 총 15번
        print(sSize)
        # sSize * 6이 시작 위치
        resizeFrame = frame[sSize * 6:windowSize + sSize * 6]