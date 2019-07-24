import pandas as pd
import glob

if __name__ == '__main__':
    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\경제지표'
    allFiles = glob.glob(path + '/*.csv')
    frame = pd.read_csv('^KS11.csv', encoding='CP949')
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