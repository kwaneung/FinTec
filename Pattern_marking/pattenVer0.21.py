import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

class PatternMaker(object):
    """
    function
    1. 패턴 시그널 컬럼 생성: classify_set
    2. N일간 다양한 등락률 컬럼 생성: add_ratio_culumn

    인스턴스 생성할 시 param: dataframe: 사용될 지수 데이터프레임,
                 selection: 어떤 패턴시그널을 선택할지 정하는 변수 ex)0101010 방식으로 사용 (현재 0b를 붙여함 안붙이도록 수정할 예정)
                 optiondic: 각 패턴에 사용될 옵션 딕셔너리를 가지고있는 딕셔너리

    """
    HAMMER = 0b1000000000
    UPINCLUDE = 0b0100000000
    THREEUP = 0b0010000000
    WUP = 0b0001000000
    SHARPUP = 0b0000100000
    METEOR = 0b0000010000
    DOWNINCLUDE = 0b0000001000
    THREEDOWN = 0b0000000100
    SHARPDOWN = 0b0000000010
    MDOWN = 0b0000000001
    allPatternList=[HAMMER, UPINCLUDE, THREEUP, WUP, SHARPUP, METEOR, DOWNINCLUDE, THREEDOWN, SHARPDOWN, MDOWN]

    def __init__(self, dataframe, set, optiondic):
        # 클래스 초기화
        self.dataframe = dataframe
        self.set = set
        self.optiondic = optiondic

    def set_pattern(self):
        # 파라미터로 받은 실행할 패턴에 대한 2진수 구별
        for bit in self.allPatternList:
            runningPattern = self.set & bit
            if runningPattern is not 0:
                self.classify_set(runningPattern)

    def classify_set(self, pattern):
        # 구별한 패턴 실행
        if pattern is self.HAMMER:
            self.make_hammer_sig(self.dataframe, self.optiondic['hammer'])
        elif pattern is self.UPINCLUDE:
            self.make_upinclude_sig(self.dataframe, self.optiondic['upinclude'])
        elif pattern is self.THREEUP:
            self.make_threeup_sig(self.dataframe)
        elif pattern is self.WUP:
            self.make_wup_sig(self.dataframe, self.optiondic['wup'])
        elif pattern is self.SHARPUP:
            self.make_sharpup_sig(self.dataframe, self.optiondic['sharpup'])
        elif pattern is self.METEOR:
            self.make_meteor_sig(self.dataframe, self.optiondic['meteor'])
        elif pattern is self.DOWNINCLUDE:
            self.make_downinclude_sig(self.dataframe, self.optiondic['downinclude'])
        elif pattern is self.THREEDOWN:
            self.make_threedown_sig(self.dataframe)
        elif pattern is self.SHARPDOWN:
            self.make_sharpdown_sig(self.dataframe, self.optiondic['sharpdown'])
        elif pattern is self.MDOWN:
            self.make_mdown_sig(self.dataframe, self.optiondic['mdown'])
    def add_ratio_culumn(self):
        """
        다음날/10/일 상승/하락비율컬럼
        다음날/5/10일간 출현일종가 대비 평균증감율, 최고상승률, 최저하락률 컬럼 생성 함수
        :return:
        """
        self.make_Ndayafter_rate(self.dataframe)
        self.make_Ndayavg_rate(self.dataframe)
        self.make_NdayHigh_rate(self.dataframe)
        self.make_NdayLow_rate(self.dataframe)

    def make_hammer_sig(self, dataframe, option):  # 망치형
        # option = {
        #           'unit': 유닛크기 default = 1
        #           'maxHeadRatio': 헤드대비 스틱 배율 head over stick default = 1
        #           'minHSizeUnit': 최소 헤드 크기 num of unit default = 1
        #             }
        # headSize = unit * minHSizeUnit

        dataframe['hammer_sig'] = np.where((dataframe.Close > dataframe.Open)  # 양봉 해머의 경우
                                & ((dataframe.Open - dataframe.Low) >= (
                            (dataframe.Close - dataframe.Open) * option['maxHeadRatio']))  # 스틱의 크기가 헤드의 maxHeadRatio 배수이상인경우
                                & ((dataframe.Close - dataframe.Open) >= option['minHSizeUnit'])  # 헤드의 크기가 minHSizeUnit 이상인 경우
                                & (dataframe.Close == dataframe.High), 1, 0)  # 고가와 종가가 같은 경우

        return dataframe


    def make_upinclude_sig(self, dataframe, option):  # 상승 장악형
        # option = {
        #           minUpRate : 종가/시가 - 1 default = 0.01
        #          }

        dataframe['upinclude_sig'] = np.where((dataframe.Close > dataframe.Open)  # 오늘 양봉
                                   & (dataframe.Close.shift(+1) < dataframe.Open.shift(+1))  # 어제 음봉
                                   & (dataframe.Close >= dataframe.Open.shift(+1))  # 어제의 시가보다 오늘의 종가가 크고
                                   & (dataframe.Close.shift(+1) >= dataframe.Open)  # 오늘의 시가가 어제의 종가보다 작음 -> 어제가 오늘에 포함
                                   & ((dataframe.Close / dataframe.Open - 1) >= option['minUpRate']), 1,
                                   0)  # 캔들의 아래인 시가대비 캔들의 위쪽인 종가 상승률

        return dataframe

    def make_threeup_sig(self, dataframe):
        """
        적삼병 threeUp
        양봉 3개에 이전보다 저점 고점 종가 상승
        옵션 x
        threeUpLonger threeUpShorter (시가 종가)
        """
        dataframe['threeUp_sig'] = np.where(((dataframe['Close'] > dataframe['Open'])
                                            & (dataframe['Close'].shift(1) > dataframe['Open'].shift(1))
                                            & (dataframe['Close'].shift(2) > dataframe['Open'].shift(2))
                                            & ((dataframe.Low - dataframe.Low.shift(1)) > 0)
                                            & ((dataframe.High - dataframe.High.shift(1)) > 0)
                                            & ((dataframe.Close - dataframe.Close.shift(1)) > 0)
                                            & ((dataframe.Low.shift(1) - dataframe.Low.shift(2)) > 0)
                                            & ((dataframe.High.shift(1) - dataframe.High.shift(2)) > 0)
                                            & ((dataframe.Close.shift(1) - dataframe.Close.shift(2)) > 0)
                                            ), 1, 0)
        dataframe['threeUpLonger_sig'] = np.where(((dataframe['Close'] > dataframe['Open'])
                                             & (dataframe['Close'].shift(1) > dataframe['Open'].shift(1))
                                             & (dataframe['Close'].shift(2) > dataframe['Open'].shift(2))
                                             & ((dataframe.Low - dataframe.Low.shift(1)) > 0)
                                             & ((dataframe.High - dataframe.High.shift(1)) > 0)
                                             & ((dataframe.Close - dataframe.Close.shift(1)) > 0)
                                             & ((dataframe.Low.shift(1) - dataframe.Low.shift(2)) > 0)
                                             & ((dataframe.High.shift(1) - dataframe.High.shift(2)) > 0)
                                             & ((dataframe.Close.shift(1) - dataframe.Close.shift(2)) > 0)
                                             & ((dataframe.Close.shift(1) - dataframe.Open.shift(1)) > (dataframe.Close.shift(2) - dataframe.Open.shift(2)))
                                             & ((dataframe.Close - dataframe.Open) > (dataframe.Close.shift(1) - dataframe.Open.shift(1)))
                                                ), 1, 0)
        dataframe['threeUpShorter_sig'] = np.where(((dataframe['Close'] > dataframe['Open'])
                                                   & (dataframe['Close'].shift(1) > dataframe['Open'].shift(1))
                                                   & (dataframe['Close'].shift(2) > dataframe['Open'].shift(2))
                                                   & ((dataframe.Low - dataframe.Low.shift(1)) > 0)
                                                   & ((dataframe.High - dataframe.High.shift(1)) > 0)
                                                   & ((dataframe.Close - dataframe.Close.shift(1)) > 0)
                                                   & ((dataframe.Low.shift(1) - dataframe.Low.shift(2)) > 0)
                                                   & ((dataframe.High.shift(1) - dataframe.High.shift(2)) > 0)
                                                   & ((dataframe.Close.shift(1) - dataframe.Close.shift(2)) > 0)
                                                   & ((dataframe.Close.shift(1) - dataframe.Open.shift(1)) < (
                                                        dataframe.Close.shift(2) - dataframe.Open.shift(2)))
                                                   & ((dataframe.Close - dataframe.Open) < (
                                                        dataframe.Close.shift(1) - dataframe.Open.shift(1)))
                                                   ), 1, 0)

    def make_threedown_sig(self, dataframe):
        """
        흑삼병 threeUp
        양봉 3개에 이전보다 저점 고점 종가 하락
        옵션 x
        threeDownLonger threeDownShorter (시가 종가)
        """
        dataframe['threeDown_sig'] = np.where(((dataframe['Close'] < dataframe['Open'])
                                            & (dataframe['Close'].shift(1) < dataframe['Open'].shift(1))
                                            & (dataframe['Close'].shift(2) < dataframe['Open'].shift(2))
                                            & ((dataframe.Low.shift(1) - dataframe.Low) > 0)
                                            & ((dataframe.High.shift(1) - dataframe.High) > 0)
                                            & ((dataframe.Close.shift(1) - dataframe.Close) > 0)
                                            & ((dataframe.Low.shift(2) - dataframe.Low.shift(1)) > 0)
                                            & ((dataframe.High.shift(2) - dataframe.High.shift(1)) > 0)
                                            & ((dataframe.Close.shift(2) - dataframe.Close.shift(1)) > 0)
                                            ), 1, 0)

        dataframe['threeDownLonger_sig'] = np.where(((dataframe['Close'] < dataframe['Open'])
                                            & (dataframe['Close'].shift(1) < dataframe['Open'].shift(1))
                                            & (dataframe['Close'].shift(2) < dataframe['Open'].shift(2))
                                            & ((dataframe.Low.shift(1) - dataframe.Low) > 0)
                                            & ((dataframe.High.shift(1) - dataframe.High) > 0)
                                            & ((dataframe.Close.shift(1) - dataframe.Close) > 0)
                                            & ((dataframe.Low.shift(2) - dataframe.Low.shift(1)) > 0)
                                            & ((dataframe.High.shift(2) - dataframe.High.shift(1)) > 0)
                                            & ((dataframe.Close.shift(2) - dataframe.Close.shift(1)) > 0)
                                            & ((dataframe.Open.shift(1) - dataframe.Close.shift(1)) > (
                                                dataframe.Open.shift(2) - dataframe.Close.shift(2)))
                                            & ((dataframe.Open - dataframe.Close) > (
                                                dataframe.Open.shift(1) - dataframe.Close.shift(1)))
                                            ), 1, 0)

        dataframe['threeDownShorter_sig'] = np.where(((dataframe['Close'] < dataframe['Open'])
                                                   & (dataframe['Close'].shift(1) < dataframe['Open'].shift(1))
                                                   & (dataframe['Close'].shift(2) < dataframe['Open'].shift(2))
                                                   & ((dataframe.Low.shift(1) - dataframe.Low) > 0)
                                                   & ((dataframe.High.shift(1) - dataframe.High) > 0)
                                                   & ((dataframe.Close.shift(1) - dataframe.Close) > 0)
                                                   & ((dataframe.Low.shift(2) - dataframe.Low.shift(1)) > 0)
                                                   & ((dataframe.High.shift(2) - dataframe.High.shift(1)) > 0)
                                                   & ((dataframe.Close.shift(2) - dataframe.Close.shift(1)) > 0)
                                                   & ((dataframe.Open.shift(1) - dataframe.Close.shift(1)) < (
                                                        dataframe.Open.shift(2) - dataframe.Close.shift(2)))
                                                   & ((dataframe.Open - dataframe.Close) < (
                                                        dataframe.Open.shift(1) - dataframe.Close.shift(1)))
                                                   ), 1, 0)


    def make_meteor_sig(self, dataframe, option):  # 유성형
        # option = {
        #           'unit': 유닛크기 default = 1
        #           'maxHeadRatio': 헤드대비 스틱 배율 head over stick default = 1
        #           'minHSizeUnit': 최소 헤드 크기 num of unit default = 1
        #           }
        # headSize = unit * minHSizeUnit

        dataframe['meteor_sig'] = np.where((dataframe.Close < dataframe.Open)  # 양봉 해머의 경우
                                & ((dataframe.High - dataframe.Open) >= (
                    (dataframe.Open - dataframe.Low) * option['maxHeadRatio']))  # 스틱의 크기가 헤드의 maxHeadRatio 배수이상인경우
                                & ((dataframe.Open - dataframe.Close) >= option['minHSizeUnit'])  # 헤드의 크기가 minHSizeUnit 이상인 경우
                                & (dataframe.Close == dataframe.Low), 1, 0)  # 고가와 종가가 같은 경우

        return dataframe

    def make_downinclude_sig(self, dataframe, option):  # 하락 장악형
        # option = {
        #           minUpRate : 시가/종가 - 1 default = 0.01
        #          }

        dataframe['downinclude_sig'] = np.where((dataframe.Close < dataframe.Open)  # 오늘 음봉
                                   & (dataframe.Close.shift(+1) > dataframe.Open.shift(+1))  # 어제 양봉
                                   & (dataframe.Close.shift(+1) <= dataframe.Open)  # 어제의 종가보다 오늘의 시가가 크고
                                   & (dataframe.Close <= dataframe.Open.shift(+1))  # 오늘의 종가가 어제의 시가보다 작음 -> 어제가 오늘에 포함
                                   & ((dataframe.Open / dataframe.Close) - 1 >= option['minUpRate']), 1,
                                   0)  # 캔들의 아래인 종가대비 캔들의 위쪽인 시가 상승률

        return dataframe

    def make_sharpup_sig(self, dataframe, option):
        # Sharpup 패턴 시그널 생성 함수
        # option = {
        #           ratio : 상승,하락비율
        #         }

        dataframe['sharpup_sig'] = np.where((dataframe.Close / dataframe.Close.shift(+1)) - 1 > option['ratio'], 1, 0)

        return dataframe

    def make_sharpdown_sig(self, dataframe, option):
        # Sharpdown 패턴 시그널 생성 함수
        # option = {
        #           ratio : 상승,하락비율
        #         }

        dataframe['sharpown_sig'] = np.where((dataframe.Close / dataframe.Close.shift(+1)) - 1 < option['ratio'], 1, 0)

        return dataframe

    def make_wup_sig(self, dataframe, option):
        # Wup 패턴 시그널 생성 함수
        # option = {
        #           ma_value : 이동평균 일수
        #         }

        option.update({'ma_value': str(option['ma_value'])})

        dataframe['ma' + option['ma_value']] = (dataframe.Close.rolling(int(option['ma_value'])).mean())
        dataframe['ma' + option['ma_value'] + '_l'] = np.where(
            (dataframe['ma' + option['ma_value']] > dataframe['ma' + option['ma_value']].shift(+1)) &
            (dataframe['ma' + option['ma_value']].shift(+2) > dataframe['ma' + option['ma_value']].shift(+1)), 1, 0)

        wup = [0 for i in range(0, len(dataframe))]

        for i in range(0, len(dataframe)):
            if dataframe['ma' + option['ma_value'] + '_l'][i] == 1:
                for j in range(i + 1, len(dataframe)):
                    if dataframe['ma' + option['ma_value'] + '_l'][j] == 1:
                        if dataframe['ma' + option['ma_value']][i] < dataframe['ma' + option['ma_value']][j]:
                            wup[j] = 1
                            break

        dataframe['wup'] = wup

        return dataframe

    def make_mdown_sig(self, dataframe, option):
        # Mdown 패턴 시그널 생성 함수
        # option = {
        #           ma_value : 이동평균 일수
        #         }

        option.update({'ma_value': str(option['ma_value'])})

        dataframe['ma' + option['ma_value']] = (dataframe.Close.rolling(int(option['ma_value'])).mean())
        dataframe['ma' + option['ma_value'] + '_h'] = np.where(
            (dataframe['ma' + option['ma_value']] < dataframe['ma' + option['ma_value']].shift(+1)) &
            (dataframe['ma' + option['ma_value']].shift(+2) < dataframe['ma' + option['ma_value']].shift(+1)), 1, 0)

        mdown = [0 for i in range(0, len(dataframe))]

        for i in range(0, len(dataframe)):
            if dataframe['ma' + option['ma_value'] + '_h'][i] == 1:
                for j in range(i + 1, len(dataframe)):
                    if dataframe['ma' + option['ma_value'] + '_h'][j] == 1:
                        if dataframe['ma' + option['ma_value']][i] > dataframe['ma' + option['ma_value']][j]:
                            mdown[j] = 1
                            break

        dataframe['mdown'] = mdown

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
        dataframe['5_avgRate'] = (dataframe.Close.rolling(5).mean().shift(-5) - dataframe.Close) / dataframe.Close
        dataframe['10_avgRate'] = (dataframe.Close.rolling(10).mean().shift(-10) - dataframe.Close) / dataframe.Close
        # dataframe['15_avgRate'] =  (data.Close.rolling(15).mean().shift(-14) - data.Close)/ data.Close

    def make_NdayHigh_rate(self, dataframe):
        """
        다음날/5/10일간 출현일종가 대비 최고상승률 컬럼생성
        :param dataframe:
        :return:
        """
        dataframe['next_highRate'] = (dataframe['High'].shift(-1) - dataframe['Close']) / dataframe['Close']
        dataframe['5_highRate'] = (dataframe['High'].rolling(5).max().shift(-5) - dataframe.Close) / dataframe.Close
        dataframe['10_highRate'] = (dataframe['High'].rolling(10).max().shift(-10) - dataframe.Close) / dataframe.Close
        # dataframe['15_highRate'] =  (data.Close.rolling(15).max().shift(-14) - data.Close)/ data.Close

    def make_NdayLow_rate(self, dataframe):
        """
        다음날/5/10일간 출현일종가 대비 최저하락률컬럼 생성
        :param dataframe:
        :return:
        """
        dataframe['next_highRate'] = (dataframe['Low'].shift(-1) - dataframe['Close']) / dataframe['Close']
        dataframe['5_LowRate'] = (dataframe.Low.rolling(5).min().shift(-5) - dataframe.Close) / dataframe.Close
        dataframe['10_LowRate'] = (dataframe.Low.rolling(10).min().shift(-10) - dataframe.Close) / dataframe.Close
        # dataframe['15_LowRate'] = (data.Close.rolling(15).min().shift(-14) - data.Close)/ data.Close

if __name__ == "__main__":
    kospidata = pd.read_csv('KS11.csv') # KOSPI에 대해서 분석
    kospidata = kospidata.fillna(method='ffill')
    """
    hammer_option = {
        'unit':1,
        'maxHeadRatio':1,
        'minHSizeUnit':1
    }
    upinclude_option={
        'minUpRate':0.01
    }
    sharpupoption = {
        'ratio' : 1
    }
    """
    optiondic = {'hammer':{'unit':1, 'maxHeadRatio':1, 'minHSizeUnit':1},
                 'upinclude':{'minUpRate':0.01},
                 'sharpup':{'ratio' : 1},
                 'meteor': {'unit':1, 'maxHeadRatio':1, 'minHSizeUnit':1},
                 'downinclude': {'minUpRate':0.01},
                 'sharpdown': { 'ratio' : 1},
                 'wup':{'ma_value': 3},
                 'mdown':{'ma_value': 3}
                 }

    pt = PatternMaker(kospidata, 0b11111111111, optiondic)
    pt.add_ratio_culumn()  # N일간 다양한 등락률 컬럼 생성
    pt.set_pattern() # 패턴 시그널 컬럼 생성
    pt.dataframe.to_csv('result.csv')

    print(pt.dataframe)
