import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn import datasets
import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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
    dfx = frame[feature]
    dfy = frame[["cm5up"]]
    # print(dfx.shape)
    # print(dfy.shape)

    X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_test = frame[feature].iloc[292:, :]
    X_train = frame[feature].iloc[:292, :]
    y_train = frame['cm5up'].iloc[:292]
    y_test = frame['cm5up'].iloc[292:]

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test_std)
    print('총 테스트 개수 : %d, 오류 개수 : %d' % (len(y_test), (y_test != y_pred).sum()))
    print('정확도 : %.2f' % accuracy_score(y_test, y_pred))

    iris = datasets.load_iris()
    dot_data = export_graphviz(tree, out_file=None, feature_names=feature_cols, class_names=['0', '5% JUMP'], filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # (graph,) = pydot.graph_from_dot_file('DT.dot', encoding='utf8')
    graph.write_png('decision_trees.png')