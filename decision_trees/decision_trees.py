from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn import datasets
import pydotplus
from linear_algebra import dot
import random, math
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import glob
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def logistic(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))

if __name__ == '__main__':

    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\decision_trees\종속변수\1차 지표'
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

    dow = pd.read_csv('Dow-Monthly.csv', encoding='CP949')

    frame = pd.merge(frame, dow, on='DATE')
    frame = frame.dropna()
    frame = frame.set_index('DATE')
    # frame = frame.sort_values('DATE')

    feature = ["미국 내구재 주문", "미국 소비자 물가 상승률", "미국 소비율"]
    feature_cols = ["US Durable Goods Order", "US consumer price inflation", "US consumption rate"]
    Dependent = 'cm5down'
    dfx = frame[feature]
    dfy = frame[[Dependent]]
    # print(dfx.shape)
    # print(dfy.shape)

    # X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3)

    # sc = StandardScaler()
    # sc.fit(X_train)
    X_test = frame[feature].iloc[292:, :]
    X_train = frame[feature].iloc[:292, :]
    y_train = frame[Dependent].iloc[:292]
    y_test = frame[Dependent].iloc[292:]

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test_std)
    print('총 테스트 개수 : %d, 오류 개수 : %d' % (len(y_test), (y_test != y_pred).sum()))
    print('정확도 : %.2f' % accuracy_score(y_test, y_pred))

    iris = datasets.load_iris()
    dot_data = export_graphviz(tree, out_file=None, feature_names=feature_cols, class_names=['0', Dependent], filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # (graph,) = pydot.graph_from_dot_file('DT.dot', encoding='utf8')
    graph.write_png('decision_trees.png')

    print()
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    random.seed(0)

    myreg = LinearRegression(False).fit(X_train, y_train)

    true_positives = false_positives = true_negatives = false_negatives = 0
    p_list = []

    for x_i, y_i in zip(X_test, y_test):
        predict = logistic(dot(myreg.coef_, x_i))
        p_list.append(1 if predict >= 0.5 else 0)

        if y_i == 1 and predict >= 0.5:  # TP: paid and we predict paid
            true_positives += 1
        elif y_i == 1:  # FN: paid and we predict unpaid
            false_negatives += 1
        elif predict >= 0.5:  # FP: unpaid and we predict paid
            false_positives += 1
        else:  # TN: unpaid and we predict unpaid
            true_negatives += 1

    print("true_positives : " + str(true_positives))  # 실제 True인 정답을 True라고 예측 (정답)
    print("false_negatives : " + str(false_negatives))  # 실제 True인 정답을 False라고 예측 (오답)
    print("false_positives : " + str(false_positives))  # 실제 False인 정답을 True라고 예측 (오답)
    print("true_negatives : " + str(true_negatives))  # 실제 False인 정답을 False라고 예측 (정답)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print("precision", precision)  # 정밀도. True라고 분류한 것 중에서 실제 True인 것의 비율
    print("recall", recall)  # 재현율. 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
