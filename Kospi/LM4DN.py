import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import glob

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

    LM4DN = pd.read_csv('LM4DN.csv', encoding='CP949')
    frame = pd.merge(frame, LM4DN, on='DATE')

    # kospi = pd.read_csv('Adj Close.csv', encoding='CP949')
    # frame = pd.merge(frame, kospi, on='DATE')

    frame = frame.dropna()
    frame = frame.set_index('DATE')
    # frame = frame[["HM4UP", "Adj Close", "국제유가", "금시세", "무역 가중치 미국 달러 인덱스", "무역 가중치 미국 달러 인덱스", "미국 ISM 제조업 구매자지수", "미국 내구재 주문", "미국 비농업 고용자수",
    #          "미국 비농업 급여", "미국 생산 자원사용률", "미국 소비율", "미국 소비자 물가 상승률", "미국 소비자 신뢰지수", "미국 수입증가율",
    #          "미국 신규 실업수당 청구건수", "미국 신규주택착공", "미국 신규주택판매 지수",
    #          "미국 실업률", "미국 장단기 스프레드", "미국 항공기 제외, 비국방 자본재 주문", "미국의 소매판매지수",
    #          "미연준 기준금리", "중국 산업생산 YOY", "필라델피아 연준 제조업지수"]]

    corr = frame.corr(method='pearson')
    corr.to_csv('corr.csv', encoding='CP949')

    feature = ["미국 ISM 제조업 구매자지수", "미국 소비자 신뢰지수", "미국의 소매판매지수"]
    frame[feature] = frame[feature].apply(lambda x: x.astype(float))
    Dependent = 'LM4DN'
    x = frame[feature]
    y = frame[[Dependent]]
    print(len(frame))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_test = frame[feature].iloc[88:, :]
    x_train = frame[feature].iloc[:88, :]
    y_train = frame[Dependent].iloc[:88]
    y_test = frame[Dependent].iloc[88:]
    y_test.to_csv("y_test.csv", encoding="cp949")

    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    print(len(y_test))

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    x2 = sm.add_constant(x)
    model = sm.OLS(y, x2)
    result = model.fit()
    print(result.summary())

    y_pred = log_reg.predict(x_test)
    print(y_pred)
    print(list(y_test))

    print('정확도 :', metrics.accuracy_score(y_test, y_pred))

    true_positives = false_positives = true_negatives = false_negatives = 0
    dff = pd.DataFrame()
    i = 0
    y_test = y_test.values
    for x_i, y_i in zip(y_pred, y_test):
        if y_i == 1 and x_i == 1:  # TP: paid and we predict paid
            true_positives += 1
            dff.loc[i, 'TP'] = 1
        elif y_i == 1:  # FN: paid and we predict unpaid
            false_negatives += 1
            dff.loc[i, 'FN'] = 1
        elif x_i == 1:  # FP: unpaid and we predict paid
            false_positives += 1
            dff.loc[i, 'FP'] = 1
        else:  # TN: unpaid and we predict unpaid
            true_negatives += 1
            dff.loc[i, 'TN'] = 1
        i = i + 1

    dff.to_csv("y_pred.csv", encoding="cp949")

    print("true_positives : %d" %true_positives) # 실제 True인 정답을 True라고 예측 (정답)
    print("false_negatives : %d" %false_negatives) # 실제 True인 정답을 False라고 예측 (오답)
    print( "false_positives : %d" %false_positives) # 실제 False인 정답을 True라고 예측 (오답)
    print("true_negatives : %d" %true_negatives) # 실제 False인 정답을 False라고 예측 (정답)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print("precision", precision)
    print("recall", recall)

