import pandas as pd

# 한달에 여러번 갱신될 경우 월1개의 데이터평균으로 산출
if __name__ == "__main__":
    df = pd.read_csv('^KS11-Monthly.csv', encoding='CP949')
    df = df.sort_values('DATE')
    year = 2008
    month = 12
    sum = 0  # 합친 데이터 값
    cnt = 0  # 합친 일 수

    for i in range(len(df.values)):
        if month < 10:
            if df.loc[i, 'DATE'][0:7] != str(year) + '-0' + str(month):
                df.loc[i, 'NEWDATE'] = str(year) + '-' + str(month) + '-01'
                df.loc[i, 'HM4UP2'] = sum / cnt
                sum = 0
                cnt = 0
                if month == 12:
                    month = 1
                    year = year + 1
                else:
                    month = month + 1
            # else:
            sum = sum + df.loc[i][25]
            cnt = cnt + 1

        else:
            if df.loc[i, 'DATE'][0:7] != str(year) + '-' + str(month):
                df.loc[i, 'NEWDATE'] = str(year) + '-' + str(month) + '-01'
                df.loc[i, 'HM4UP2'] = sum / cnt
                sum=0
                cnt = 0
                if month == 12:
                    month = 1
                    year = year + 1
                else:
                    month = month + 1
            # else:
            sum = sum + df.loc[i][25]
            cnt = cnt + 1

    df = df[["NEWDATE", "HM4UP2"]]
    df = df.dropna()
    df.to_csv('HM4UP.csv', encoding='CP949')
    print(df)