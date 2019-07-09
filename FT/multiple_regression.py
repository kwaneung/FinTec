from collections import Counter
from functools import partial
from linear_algebra import dot, vector_add
from stats import median, standard_deviation
from probability import normal_cdf
from gradient_descent import minimize_stochastic
from simple_linear_regression import total_sum_of_squares
import math, random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import glob
import timeit
from sklearn.model_selection import train_test_split

start = timeit.default_timer()


def predict(x_i, beta):
    return dot(x_i, beta)


def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)


def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2


def squared_error_gradient(x_i, y_i, beta):
    """the gradient corresponding to the ith squared error term"""
    return [-2 * x_ij * error(x_i, y_i, beta)
            for x_ij in x_i]


def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error,
                               squared_error_gradient,
                               x, y,
                               beta_initial,
                               0.001)


def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2
                                for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)


def bootstrap_sample(data):
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data, stats_fn, num_samples):
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data))
            for _ in range(num_samples)]


def estimate_sample_beta(sample):
    x_sample, y_sample = list(zip(*sample))  # magic unzipping trick
    return estimate_beta(x_sample, y_sample)


def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


#
# REGULARIZED REGRESSION
#

# alpha is a *hyperparameter* controlling how harsh the penalty is
# sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta, alpha):
    return alpha * dot(beta[1:], beta[1:])


def squared_error_ridge(x_i, y_i, beta, alpha):
    """estimate error plus ridge penalty on beta"""
    return error(x_i, y_i, beta) ** 2 + ridge_penalty(beta, alpha)


def ridge_penalty_gradient(beta, alpha):
    """gradient of just the ridge penalty"""
    return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]


def squared_error_ridge_gradient(x_i, y_i, beta, alpha):
    """the gradient corresponding to the ith squared error term
    including the ridge penalty"""
    return vector_add(squared_error_gradient(x_i, y_i, beta),
                      ridge_penalty_gradient(beta, alpha))


def estimate_beta_ridge(x, y, alpha):
    """use gradient descent to fit a ridge regression
    with penalty alpha"""
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(partial(squared_error_ridge, alpha=alpha),
                               partial(squared_error_ridge_gradient,
                                       alpha=alpha),
                               x, y,
                               beta_initial,
                               0.001)


def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])


# -------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\FT\종속변수'
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
            frame = pd.merge(frame, df, on='DATE', how='outer')

    tmp = pd.read_csv('^KS11-Daily.csv', encoding='CP949')

    frame = pd.merge(frame, tmp, on='DATE', how='outer')
    frame['bias'] = 1
    frame = frame.sort_values('DATE')
    frame = frame[frame.DATE >= '1997-07-01']  # KOSPI 지수의 최저 날짜 이전 데이터 날림.

    dfx = frame[["bias", "msci emerging markets", "경상수지", "상품수지"
        , "국제유가", "금기준가격(원/g)", "무역 가중치 미국 달러 인덱스", "미국 IHS 제조업 PMI Actual"
        , "미국 ISM 제조업 구매자지수 Actual", "미국 내구재 주문", "미국 비농업 고용자수", "미국 생산 자원사용률"
        , "미국 소비율", "미국 소비자 물가 상승률", "미국 소비자 신뢰지수", "미국 수입증가율"
        , "미국 신규 실업수당 청구건수", "미국 신규주택착공", "미국 신규주택판매 지수", "미국 실업률"
        , "미국 장단기 스프레드", "미국 재고증가율", "미국 항공기 제외, 비국방 자본재 주문", "미국의 소매판매지수"
        , "미연준 기준금리", "한국은행기준 원달러환율", "은시세", "중국 산업생산 YOY Actual"
        , "중국 산업생산 전년대비증감률", "중국 주택가격 증가율", "중국 차이신 PMI Actual", "컨퍼런스보드 소비자 심리지수 Actual"
        , "필라델피아 연준 제조업지수 Actual"]]

    dfx = dfx.fillna(method='ffill')
    dfx = dfx.fillna(method='bfill')
    dfx.to_csv('dfx.csv', encoding='CP949')

    frame2 = pd.read_csv('^KS11-Daily.csv', encoding='CP949')
    frame2 = pd.merge(frame2, frame, on='DATE', how='outer')
    dfy = frame[["Close"]]
    dfy = dfy.fillna(method='ffill')
    dfy = dfy.fillna(method='bfill')
    dfy.to_csv('dfy.csv', encoding='CP949')

    dfx = dfx.values
    dfy = dfy.values
    dfy = np.ravel(dfy, order='C')  # 1차원 리스트로 변환

    X_train, X_test, Y_train, Y_test = train_test_split(dfx, dfy, test_size=0.20, random_state=321)  # 학습 데이터 : 테스트 데이터 = 8 : 2

    random.seed(0)

    myreg = LinearRegression(False).fit(dfx, dfy)
    print(myreg.coef_)

    stop = timeit.default_timer()
    print(stop - start)
