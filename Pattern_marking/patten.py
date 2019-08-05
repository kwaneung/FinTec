import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

class Patternmaker(object):
    """
    기능
    1. 패턴 시그널 생성: select_pattern
    2. N일간 다양한 등락률 생성: add_ratio

    인스턴스 생성할 시 param: dataframe: 사용될 지수 데이터프레임,
                 selection: 어떤 패턴시그널을 선택할지 정하는 변수 ex)0101010 방식으로 사용 (현재 0b를 붙여함 안붙이도록 수정할 예정)
                 optiondic: 각 패턴에 사용될 옵션 딕셔너리를 가지고있는 딕셔너리


    """
    hammer = 0b1000000
    upinclude = 0b0100000
    threeup = 0b0010000
    Wup = 0b0001000
    sharpup = 0b0000100

    uppattern = [hammer, upinclude, threeup, Wup, sharpup]

    def __init__(self, dataframe, selection, optiondic):
        # 클래스 초기화
        self.dataframe = dataframe
        self.selection = selection
        self.optiondic = optiondic

    def classify_selection(self):

        # 파라미터로 받은 실행할 패턴에 대한 2진수 구별

        for i in self.uppattern:
            sel = self.selection & i
            if sel is not 0:
                self.select_pattern(sel)

    def select_pattern(self, pattern):
        # 구별된 패턴 실행
        if pattern is self.hammer:
            self.make_hammer_sig(self.dataframe, self.optiondic['hammer'])
        if pattern is self.upinclude:
            self.make_upinclude_sig(self.dataframe, self.optiondic['upinclude'])
        if pattern is self.threeup:
            print('threeup')
        if pattern is self.Wup:
            print('Wup')
        if pattern is self.sharpup:
            print('sharpup')

    def add_ratio(self):
        """
        다음날/10/일 상승/하락비율컬럼
        다음날/5/10일간 출현일종가 대비 평균증감율, 최고상승률, 최저하락률 컬럼 생성 함수
        :return:
        """
        self.make_Ndayafter_rate(self.dataframe)
        self.make_Ndayavg_rate(self.dataframe)
        self.make_NdayHigh_rate(self.dataframe)
        self.make_NdayLow_rate(self.dataframe)

    def make_hammer_sig(self, dataframe, option):
        # 해머 패턴 시그널 생성 함수
        # option = {
        #            headlen:최소 헤드길이
        #            multiple:헤드와 막대길이 몇 배수 할지
        #            limit:헤드위에 막대 길이를 얼마나 줄지 default=0
        #         }
        dataframe['hammer_sig'] = np.where((abs(dataframe.Close-dataframe.Open)*option['multiple']<=abs(dataframe.Open-dataframe.Low))
                                            &(abs(dataframe.Close-dataframe.High)<=option['limit'])
                                           &(abs(dataframe.Close-dataframe.Open)>=option['headlen']), 1,0)
        dataframe.to_csv("test.csv")
        return dataframe

    def make_upinclude_sig(self, dataframe, option):
        # upinclude패턴 시그널 생성 함수
        # option = {
        #
        #            multiple:전날 대비 캔들이 몇배이상 커질지 선택
        #            limit:캔들 위아래에 꼬리를 얼마나 허용할지 선택 default=0
        #         }
        dataframe['upinclude_sig'] = np.where((abs(dataframe['Close'].shift(1)-dataframe['Open'].shift(1))*option['multiple']<abs(dataframe['Close']-dataframe['Open']))
                                           &(dataframe['Close'].shift(1)<dataframe['Open'].shift(1))
                                           &((dataframe.High-dataframe.Close)<=option['limit'])
                                           &((dataframe.Open-dataframe.Low)<=option['limit']), 1,0)
        #&(abs(dataframe.Close-dataframe.Open)>=option['headlen'])
        return dataframe

    def make_meteor_sig(self, dataframe, option):
        # meteor 패턴 시그널 생성 함수
        # option = {
        #            headlen:최소 헤드길이
        #            multiple:헤드와 막대길이 몇 배수 할지
        #            limit:헤드위에 막대 길이를 얼마나 줄지 default=0
        #         }
        dataframe['meteor_sig'] = np.where(
            ((dataframe.Open - dataframe.Close) * option['multiple'] <= (dataframe.High - dataframe.Open))
            & ((dataframe.Close - dataframe.Low) <= option['limit'])
            & ((dataframe.Open - dataframe.Close) >= option['headlen']), 1, 0)
        return dataframe

    def make_downinclude_sig(self, dataframe, option):
        """
        downinclude 패턴 시그널 생성 함수
        option = {
                    multiple:전날 대비 캔들이 몇배이상 커질지 선택
                    limit:캔들 위아래에 꼬리를 얼마나 허용할지 선택 default=0
                 }
         """
        dataframe['downinclude_sig'] = np.where(((dataframe['Close'].shift(1)-dataframe['Open'].shift(1))*option['multiple']<=(dataframe['Open']-dataframe['Close']))
                                           &(dataframe['Close'].shift(1)>dataframe['Open'].shift(1))
                                           &((dataframe.Close-dataframe.Low)<=option['limit'])
                                           &((dataframe.High-dataframe.Open)<=option['limit']), 1,0)
        #&(abs(dataframe.Close-dataframe.Open)>=option['headlen'])
        return dataframe

    def make_Ndayafter_rate(self, dataframe):
        """
        다음날, 5일, 10일 이후의 날과의 등락률 컬럼 생성함수
        """
        dataframe['next_updown'] = (dataframe['Close'].shift(-1) - dataframe['Close']) / dataframe['Close']  # 다음날
        dataframe['5_updown'] = (dataframe['Close'].shift(-5) - dataframe['Close']) / dataframe['Close']  # 영업일 5일 이후
        dataframe['10_updown'] = (dataframe['Close'].shift(-10) - dataframe['Close']) / dataframe['Close']  # 영업일 10일 이후

    def make_Ndayavg_rate(self, dataframe):
        """
        5/10일간 출현일종가 대비 평균증감률 컬럼생성
        다음날은 next_updown과 차이가 없어 생략
        """
        dataframe['5_avgRate'] = (dataframe.Close.rolling(5).mean().shift(-4) - dataframe.Close) / dataframe.Close
        dataframe['10_avgRate'] = (dataframe.Close.rolling(10).mean().shift(-9) - dataframe.Close) / dataframe.Close
        # dataframe['15_avgRate'] =  (data.Close.rolling(15).mean().shift(-14) - data.Close)/ data.Close

    def make_NdayHigh_rate(self, dataframe):
        """
        다음날/5/10일간 출현일종가 대비 최고상승률 컬럼생성
        :param dataframe:
        :return:
        """
        dataframe['next_highRate'] = (dataframe['High'].shift(-1) - dataframe['Close']) / dataframe['Close']
        dataframe['5_highRate'] = (dataframe.Close.rolling(5).max().shift(-4) - dataframe.Close) / dataframe.Close
        dataframe['10_highRate'] = (dataframe.Close.rolling(10).max().shift(-9) - dataframe.Close) / dataframe.Close
        # dataframe['15_highRate'] =  (data.Close.rolling(15).max().shift(-14) - data.Close)/ data.Close

    def make_NdayLow_rate(self, dataframe):
        """
        다음날/5/10일간 출현일종가 대비 최저하락률컬럼 생성
        :param dataframe:
        :return:
        """
        dataframe['next_highRate'] = (dataframe['Low'].shift(-1) - dataframe['Close']) / dataframe['Close']
        dataframe['5_LowRate'] = (dataframe.Close.rolling(5).min().shift(-4) - dataframe.Close) / dataframe.Close
        dataframe['10_LowRate'] = (dataframe.Close.rolling(10).min().shift(-9) - dataframe.Close) / dataframe.Close
        # dataframe['15_LowRate'] = (data.Close.rolling(15).min().shift(-14) - data.Close)/ data.Close


kospidata = pd.read_csv('^KS11.csv') # KOSPI에 대해서 분석
kospidata = kospidata.fillna(method='ffill')
hammer_option = {
    'headlen':3,
    'multiple':2,
    'limit':1
}
upinclude_option={
    'multiple':2,
    'limit':0
}
optiondic = {'hammer':hammer_option, 'upinclude':upinclude_option}
pt = Patternmaker(kospidata, 0b1111000, optiondic)
pt.add_ratio()
pt.classify_selection()

print(pt.dataframe)
