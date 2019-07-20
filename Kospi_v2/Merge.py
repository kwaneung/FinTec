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
    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\Kospi_v2\독립변수'
    allFiles = glob.glob(path + '/*.csv')
    frame = pd.read_csv('^KS11-Monthly.csv', encoding='CP949')
    list_ = []
    cnt = 0

    for file_ in allFiles:
        print("read " + file_)
        df = pd.read_csv(file_, encoding='CP949')
        frame = pd.merge(frame, df, on='DATE')

    frame = frame.dropna()
    frame = frame.set_index('DATE')
    frame.to_csv("KOSPI_FRAME.csv", encoding='CP949')

    corr = frame.corr(method='pearson')  # 상관계수
    corr.to_csv('corr.csv', encoding='CP949')