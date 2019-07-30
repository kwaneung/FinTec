import pandas as pd


def Mark_Hammer_Pattern(df):
    df['Hammer'] = 0

    for i in range(len(df)):
        open = df.loc[i, 'Open']
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        close = df.loc[i, 'Close']
        head = 0
        stick = 0

        if close > open and close > low:  # 양봉 해머의 경우
            head = close - open  # 헤드의 길이
            stick = open - low
            if stick >= (head * 2) and head >= 3:
                df.loc[i, 'Hammer'] = 1
        elif close < open < high:  # 음봉 해머의 경우
            head = open - close
            stick = high - open
            if stick >= (head * 2) and head >= 3:
                df.loc[i, 'Hammer'] = -1

    return df


if __name__ == '__main__':
    frame = pd.read_csv('^KS11.csv', encoding='CP949')
    marked_frame = Mark_Hammer_Pattern(frame)
    print(marked_frame['Hammer'])
    marked_frame.to_csv("mk_hammer.csv", encoding='cp949')