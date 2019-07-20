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
from sklearn.model_selection import train_test_split


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
    beta_initial = [random.random() for _ in x[0]]
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
    beta_initial = [random.random() for _ in x[0]]
    return minimize_stochastic(partial(squared_error_ridge, alpha=alpha),
                               partial(squared_error_ridge_gradient,
                                       alpha=alpha),
                               x, y,
                               beta_initial,
                               0.001)


def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    path = r'C:\Users\kwaneung\Documents\GitHub\FinTec\FinTec\FT\종속변수'
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
    frame = frame.sort_values('DATE')
    # frame = frame[frame.DATE >= '2009-01-01']
    frame = frame.dropna()
    frame.to_csv('frame.csv', encoding='CP949')

    # 수지 : 매년 1980~2019
    # 국제유가 : 매달 1980~2019
    # 금시세 : 매일 2009~2019
    # 무가미달인 : 매일 2014~2019
    # 미국IHS 제조업 PMI : 매달2회 2012~2019
    # 미국ISM 제조업 PMI : 매달 2007~2019
    # 미국 내구재주문 : 매달 1992~2019
    # 미국 비농업 고용자수 : 매달 1932~2019
    # 미국 자원사용률 : 매달 1967~2019
    # 미국 소비율 : 매달 1959~2019
    # 미국 소비자 물가상승률 : 매달 1947~2019
    # 미국 소비자 신뢰지수 : 매달 1952~2019
    # 미국 수입 증가율 : 매달 1985~2019
    # 미국 신규 실업수당 청구건수 : 매주 1967~2019
    # 미국 신규주택착공 : 매달 1959~2019
    # 미국 신규주택판매 : 매달 1963~2019
    # 미국 실업률 : 매달 1948~2019
    # 미국 장단기 스프레드 : 매일 2014~2019
    # 미국 재고등가율 : 분기 1947~2019
    # 미항제비국방자주 : 매달 1992~2019
    # 미국의 소매판매지수 : 매달 1973~2019
    # 미연준 기준금리 : 매일 2014~2019
    # 원달러 환율 : 매일 2003~2019
    # 은시세 : 매일 2017~2019
    # 중국 산업생산 YOY : 매달 2010~2019
    # 중국 산업생산 전년대비증감률 : 분기 1999~2018
    # 중국 주택가격 증가율 : 매달4일 2006~2018
    # 중국 차이신 PMI : 매달2일 2010~2019
    # 컨퍼런스 : 매달 2007~2017

    dfx = frame[["컨퍼런스보드 소비자 심리지수", "미국 신규 실업수당 청구건수"]]

    # dfx = dfx.fillna(method='ffill')
    # dfx = dfx.fillna(method='bfill')
    dfx.to_csv('dfx.csv', encoding='CP949')

    dfy = frame[["Close"]]

    # dfy = dfy.fillna(method='ffill')
    # dfy = dfy.fillna(method='bfill')
    dfy.to_csv('dfy.csv', encoding='CP949')

    corr = frame.corr(method='pearson')
    corr.to_csv('corr.csv', encoding='CP949')

    dfx = dfx.values
    dfy = dfy.values  # 데이터 프레임을 ndarray로
    dfy = np.ravel(dfy, order='C')  # 1차원 리스트로 변환

    print()

    X_train, X_test, Y_train, Y_test = train_test_split(dfx, dfy, test_size=0.20, random_state=321)

    random.seed(0)

    myreg = LinearRegression(False).fit(X_train, Y_train)
    print(myreg.coef_)

    print("train r_squere : ", myreg.score(X_train, Y_train))
    print("test r_squere : ", myreg.score(X_test, Y_test))

