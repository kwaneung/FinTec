import pandas as pd
from copy import copy


# def Mark_Hammer_Pattern(df):
#
#     Hammer_df = df[df.Close > df.Open]  # 양봉인 경우
#     Hammer_df = Hammer_df[Hammer_df.Close - Hammer_df.Open >= 3]  # head 크기가 3 이상인 경우
#     Hammer_df = Hammer_df[(Hammer_df.Close - Hammer_df.Open) * 2 <= (Hammer_df.Open - Hammer_df.Low)]  # stick이 head보다 2배 이상 큰경우
#     Hammer_df = Hammer_df[Hammer_df.Close == Hammer_df.High]  # 해머의 위로 stick이 튀어나오지 않는 경우
#
#     return Hammer_df
#
#
# def Mark_Upinclude_Pattern(df):
#
#     Upinclude_df = df.loc[(abs(df['Close'].shift(1) - df['Open'].shift(1)) * 2 < abs(df['Close']-df['Open']))
#                           & (df['Close'].shift(1) < df['Open'].shift(1))
#                           & (df.Close > df.Open.shift(1))
#                           & (df.Close.shift(1) > df.Open), :].copy()
#
#     return Upinclude_df
#
#
# def Mark_3UP_Pattern_ver2(df):
#
#     _3UP_df = df.loc[(df.Open < df.Close)
#                      & (df.Open.shift(1) < df.Close).shift(1)
#                      & (df.Open.shift(2) < df.Close.shift(2))  # 3일 연속 양봉인 경우
#                      & (df.Close - df.Open >= 5)
#                      & (df.Close.shift(1) - df.Open.shift(1) >= 5)
#                      & (df.Close.shift(2) - df.Open.shift(2) >= 5), :].copy()  # 3일 연속 캔들의 크기가 5 이상
#
#     return _3UP_df


def Mark_All_Pattern(dataFrame, Hammer_Op, Upinclude_op, _3UP_op, sharpUp_op):
    df = copy(dataFrame)
    df = Mark_Hammer_Pattern(df, Hammer_Op)
    df = Mark_Upinclude_Pattern(df, Upinclude_op)
    df = Mark_3UP_Pattern(df, _3UP_op)
    df = Mark_SharpUP_Pattern(df, sharpUp_op)
    df = Mark_Meteor_Pattern(df, Hammer_Op)
    df = Mark_Downinclude_Pattern(df, Upinclude_op)
    df = Mark_3DOWN_Pattern(df, _3UP_op)
    df = Mark_SharpDOWN_Pattern(df, sharpUp_op)

    return df


def Mark_Hammer_Pattern(dataFrame, optionDic):  # 망치형
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'S/H_ratio': 헤드대비 스틱 비율
    #                 'min_Head_Size': 최소 헤드 크기
    #             }
    df = copy(dataFrame)
    df['Hammer'] = 0

    for i in range(len(df)):
        open = df.loc[i, 'Open']
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        close = df.loc[i, 'Close']

        if close > open > low:  # 양봉 해머의 경우
            head = close - open
            stick = open - low
            if stick >= (head * optionDic['S/H_ratio']) and head >= optionDic['min_Head_Size'] * optionDic['unit'] and close >= high >= open:  # 스틱이 해머크기의 2배이상, 헤드가 3이상, 고가가 종가를 넘지않는 경우
                df.loc[i, 'Hammer'] = 1
    return df


def Mark_Upinclude_Pattern(dataFrame, optionDic):  # 상승 장악형
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'Y/T_ratio': 캔들 크기 비율
    #             }
    df = copy(dataFrame)
    df['Upinclude'] = 0

    for i in range(len(df) - 1):
        open = df.loc[i, 'Open']
        close = df.loc[i, 'Close']
        open_n = df.loc[i + 1, 'Open']
        close_n = df.loc[i + 1, 'Close']

        if close < open and close_n > open_n:  # 오늘이 음봉이고 내일이 양봉
            if close_n >= open and close >= open_n:  # 오늘이 내일에 포함
                if (open - close) * optionDic['Y/T_ratio'] <= (close_n - open_n):  # 오늘캔들보다 내일캔들이 두배이상 큼
                    df.loc[i + 1, 'Upinclude'] = 1

    return df


def Mark_3UP_Pattern(dataFrame, optionDic):  # 적삼병
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'min_Candle_Size': 최소 캔들 크기
    #             }
    df = copy(dataFrame)
    df['3UP'] = 0

    for i in range(len(df) - 3):
        open = df.loc[i, 'Open']
        close = df.loc[i, 'Close']

        if open < close and df.loc[i + 1, 'Open'] < df.loc[i + 1, 'Close'] and df.loc[i + 2, 'Open'] < df.loc[i + 2, 'Close']:  # 3연속 양봉인 경우.
            if close - open > optionDic['min_Candle_Size'] * optionDic['unit'] and df.loc[i + 1, 'Close'] - df.loc[i + 1, 'Open'] > optionDic['min_Candle_Size'] * optionDic['unit'] \
                    and df.loc[i + 2, 'Close'] - df.loc[i + 2, 'Open'] > optionDic['min_Candle_Size'] * optionDic['unit']:  # 상승한 3개의 캔들이 5이상인 경우
                df.loc[i + 2, '3UP'] = 1
    return df


def Mark_SharpUP_Pattern(dataFrame, optionDic):  # 급상승
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'ratio': 상승하락비율
    #             }
    df = copy(dataFrame)
    df['SharpUP'] = 0

    for i in range(len(df)):
        open = df.loc[i, 'Open']
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        close = df.loc[i, 'Close']
        if high > open and (high / open - 1) >= optionDic['ratio']:  # 시가보다 고가가 큰 경우 시가대비 고가의 상승비율이 ratio 보다 큰 경우
            df.loc[i, 'SharpUP'] = 1

    return df


def Mark_Meteor_Pattern(dataFrame, optionDic):  # 유성형
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'S/H_ratio': 헤드대비 스틱 비율
    #                 'min_Head_Size': 최소 헤드 크기
    #             }
    df = copy(dataFrame)
    df['Meteor'] = 0

    for i in range(len(df)):
        open = df.loc[i, 'Open']
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        close = df.loc[i, 'Close']

        if close < open < high:  # 음봉 해머의 경우
            head = open - close
            stick = high - open
            if stick >= (head * 2) and head >= 3 and close <= low <= open:
                df.loc[i, 'Meteor'] = 1

    return df


def Mark_Downinclude_Pattern(dataFrame, optionDic):  # 하락 장악형
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'Y/T_ratio': 캔들 크기 비율
    #             }
    df = copy(dataFrame)
    df['Downinclude'] = 0

    for i in range(len(df) - 1):
        open = df.loc[i, 'Open']
        close = df.loc[i, 'Close']
        open_n = df.loc[i + 1, 'Open']
        close_n = df.loc[i + 1, 'Close']

        if close > open and close_n < open_n:  # 오늘이 양봉이고 내일이 음봉
            if close_n <= open and close <= open_n:  # 오늘이 내일에 포함
                if (close - open) * optionDic['Y/T_ratio'] <= (open_n - close_n):  # 오늘캔들보다 내일캔들이 두배이상 큼
                    df.loc[i + 1, 'Downinclude'] = 1

    return df


def Mark_3DOWN_Pattern(dataFrame, optionDic):  # 흑삼병
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'min_Candle_Size': 최소 캔들 크기
    #             }
    df = copy(dataFrame)
    df['3DOWN'] = 0

    for i in range(len(df) - 3):
        open = df.loc[i, 'Open']
        close = df.loc[i, 'Close']

        if open > close and df.loc[i + 1, 'Open'] > df.loc[i + 1, 'Close'] and df.loc[i + 2, 'Open'] > df.loc[i + 2, 'Close']:  # 3연속 음봉인 경우.
            if open - close > optionDic['min_Candle_Size'] * optionDic['unit'] and df.loc[i + 1, 'Open'] - df.loc[i + 1, 'Close'] > optionDic['min_Candle_Size'] * optionDic['unit'] \
                    and df.loc[i + 2, 'Open'] - df.loc[i + 2, 'Close'] > optionDic['min_Candle_Size'] * optionDic['unit']:  # 상승한 3개의 캔들이 5이상인 경우
                df.loc[i + 2, '3DOWN'] = 1
    return df


def Mark_SharpDOWN_Pattern(dataFrame, optionDic):  # 급하강
    # optionDic = {
    #                 'unit': 유닛크기
    #                 'ratio': 상승하락비율
    #             }
    df = copy(dataFrame)
    df['SharpDOWN'] = 0

    for i in range(len(df)):
        open = df.loc[i, 'Open']
        high = df.loc[i, 'High']
        low = df.loc[i, 'Low']
        close = df.loc[i, 'Close']
        if low < open and abs(low / open - 1) >= optionDic['ratio']:  # 시가보다 저가가 작은 경우 시가대비 저가의 하락비율이 ratio 보다 큰 경우
            df.loc[i, 'SharpDOWN'] = 1

    return df


if __name__ == '__main__':
    frame = pd.read_csv('^KS11.csv', encoding='CP949')
    unit = 1
    # option = {'Hammer': [3 * unit, 2], 'Upinclude': 2, '3UP': 5 * unit}

    hammer_Option = {'unit': unit, 'S/H_ratio': 2, 'min_Head_Size': 3}  # 유닛크기, 헤드대비 스틱 비율, 최소 헤드 크기
    Upinclude_Option = {'unit': unit, 'Y/T_ratio': 2}  # 유닛크기, 캔들 크기 비율
    _3UP_Option = {'unit': unit, 'min_Candle_Size': 5}  # 유닛크기, 최소 캔들 크기
    sharpUp_Option = {'unit': unit, 'ratio': 0.03}  # 유닛크기, 상승하락비율

    marked_frame = Mark_All_Pattern(frame, hammer_Option, Upinclude_Option, _3UP_Option, sharpUp_Option)

    marked_frame['1days'] = marked_frame['Close'].shift(-1) / marked_frame['Close'] - 1  # 영업일 1일 이후 상승 비율
    marked_frame['5days'] = marked_frame['Close'].shift(-5) / marked_frame['Close'] - 1  # 영업일 5일 이후 상승 비율
    marked_frame['10days'] = marked_frame['Close'].shift(-10) / marked_frame['Close'] - 1  # 영업일 10일 이후 상승 비율

    marked_frame['1_avgRate'] = marked_frame['Close'].shift(-1) / marked_frame['Close'] - 1  # 1일간 평균 증감률

    marked_frame['5_avgRate'] = ((marked_frame['Close'].shift(-5) + marked_frame['Close'].shift(-4) + marked_frame['Close'].shift(-3)  # 5일간 평균 증감률
                                  + marked_frame['Close'].shift(-2) + marked_frame['Close'].shift(-1)) / marked_frame['Close'] - 5) / 5

    marked_frame['10avgRate'] = ((marked_frame['Close'].shift(-10) + marked_frame['Close'].shift(-9) + marked_frame['Close'].shift(-8)  # 10일간 평균 증감률
                                  + marked_frame['Close'].shift(-7) + marked_frame['Close'].shift(-6) + marked_frame['Close'].shift(-5)
                                  + marked_frame['Close'].shift(-4) + marked_frame['Close'].shift(-3) + marked_frame['Close'].shift(-2) + marked_frame['Close'].shift(-1)) / marked_frame['Close'] - 10) / 10

    frame_highRate = pd.DataFrame()
    func_max = lambda x: x.max()

    marked_frame['1_highRate'] = marked_frame['High'].shift(-1) / marked_frame['Close'] - 1  # 1일 이후 상승률

    for i in range(5):
        frame_highRate[str(i + 1) + '_highRate'] = marked_frame['High'].shift(-(i + 1)) / marked_frame['Close'] - 1
    marked_frame[str(i + 1) + '_highRate'] = frame_highRate.apply(func_max, axis=1)  # 5일 이후 최고 상승률

    for i in range(10):
        frame_highRate[str(i + 1) + '_highRate'] = marked_frame['High'].shift(-(i + 1)) / marked_frame['Close'] - 1
    marked_frame[str(i + 1) + '_highRate'] = frame_highRate.apply(func_max, axis=1)  # 10일 이후 최고 상승률

    frame_lowRate = pd.DataFrame()
    func_min = lambda x: x.min()

    marked_frame['1_lowRate'] = marked_frame['Low'].shift(-1) / marked_frame['Close'] - 1  # 1일 이후 하락률

    for i in range(5):
        frame_lowRate[str(i + 1) + '_lowRate'] = marked_frame['Low'].shift(-(i + 1)) / marked_frame['Close'] - 1
    marked_frame[str(i + 1) + '_lowRate'] = frame_lowRate.apply(func_min, axis=1)  # 5일 이후 최고 하락률

    for i in range(10):
        frame_lowRate[str(i + 1) + '_lowRate'] = marked_frame['Low'].shift(-(i + 1)) / marked_frame['Close'] - 1
    marked_frame[str(i + 1) + '_lowRate'] = frame_lowRate.apply(func_min, axis=1)  # 10일 이후 최고 하락률

    frame_highRate.to_csv('H_test.csv')
    frame_lowRate.to_csv('L_test.csv')

    marked_frame.to_csv("mk_All.csv", encoding='cp949')
