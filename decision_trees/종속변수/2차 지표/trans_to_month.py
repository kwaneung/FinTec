import pandas as pd

# 한달에 여러번 갱신될 경우 월1개의 데이터평균으로 산출
if __name__ == "__main__":
    df = pd.read_csv('컨퍼런스보드 소비자 심리지수.csv', encoding='CP949')
    df = df.sort_values('DATE')
    year = 2007
    month = 2
    sum = 0  # 합친 데이터 값
    cnt = 0  # 합친 일 수

    for i in range(len(df.values)):
        if month < 10:
            if df.loc[i, 'DATE'][0:7] != str(year) + '-0' + str(month):
                df.loc[i, 'NEWDATE'] = str(year) + '-' + str(month) + '-01'
                df.loc[i, '컨퍼런스보드 소비자 심리지수2'] = sum / cnt
                sum = 0
                cnt = 0
                if month == 12:
                    month = 1
                    year = year + 1
                else:
                    month = month + 1
            # else:
            sum = sum + df.loc[i][1]
            cnt = cnt + 1

        else:
            if df.loc[i, 'DATE'][0:7] != str(year) + '-' + str(month):
                df.loc[i, 'NEWDATE'] = str(year) + '-' + str(month) + '-01'
                df.loc[i, '컨퍼런스보드 소비자 심리지수2'] = sum / cnt
                cnt = 0
                if month == 12:
                    month = 1
                    year = year + 1
                else:
                    month = month + 1
            # else:
            sum = sum + df.loc[i][1]
            cnt = cnt + 1

    df = df[["NEWDATE", "컨퍼런스보드 소비자 심리지수2"]]
    df = df.dropna()
    df.to_csv('컨퍼런스보드 소비자 심리지수 월별 데이터.csv', encoding='CP949')
    print(df)