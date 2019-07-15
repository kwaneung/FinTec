from collections import Counter, defaultdict
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
import math, random

def entropy(class_probabilities):
    """given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    """find the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)

    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets )


def group_by(items, key_fn):
    """returns a defaultdict(list), where each input item
    is in the list whose key is key_fn(item)"""
    groups = defaultdict(list)
    for item in items:
        key = key_fn(item)
        groups[key].append(item)
    return groups


def partition_by(inputs, attribute):
    """returns a dict of inputs partitioned by the attribute
    each input is a pair (attribute_dict, label)"""
    return group_by(inputs, lambda x: x[0][attribute])


def partition_entropy_by(inputs,attribute):
    """computes the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


def classify(tree, input):
    """classify the input using the given decision tree"""

    # if this is a leaf node, return its value
    if tree in [True, False]:
        return tree

    # otherwise find the correct subtree
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)  # None if input is missing attribute

    if subtree_key not in subtree_dict: # if no subtree for key,
        subtree_key = None              # we'll use the None subtree

    subtree = subtree_dict[subtree_key] # choose the appropriate subtree
    return classify(subtree, input)     # and use it to classify the input

def build_tree_id3(inputs, split_candidates=None):

    # if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # print(split_candidates)
    # print()

    # count Trues and Falses in the inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0:                  # if only Falses are left
        return False                    # return a "False" leaf

    if num_falses == 0:                 # if only Trues are left
        return True                     # return a "True" leaf

    if not split_candidates:            # if no split candidates left
        return num_trues >= num_falses  # return the majority leaf

    # otherwise, split on the best attribute
    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))

    # 베스트 애트리뷰트로 파티션을 나눔. 하지만 이건 수치로 되어있기때문에 범주형으로 바꿀 필요가 있음.
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]

    # recursively build the subtrees
    subtrees = { attribute : build_tree_id3(subset, new_candidates) for attribute, subset in partitions.items() }

    subtrees[None] = num_trues > num_falses # default case

    return (best_attribute, subtrees)

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


if __name__ == "__main__":

    path = r'C:\Users\kwaneung\Desktop\17%20의사결정나무\종속변수'
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
            frame = pd.merge(frame, df, on='DATE')

    tmp = pd.read_csv('Dow-Monthly.csv', encoding='CP949')

    frame = pd.merge(frame, tmp, on='DATE')
    frame = frame.sort_values('DATE')

    dfx = frame[["미국 내구재 주문", "미국 소비자 물가 상승률", "미국 신규 실업수당 청구건수", "미국 소비율"]]
    dfy = frame[["cm5"]]

    dfx = dfx.dropna()
    tt = ()
    inputs = []

    for i in range(len(dfx.index)):
        tt = (dfx.loc[i].to_dict(), dfy.loc[i][0])
        inputs.append(tt)

    print(inputs)

    for key in ['미국 내구재 주문','미국 소비자 물가 상승률','미국 신규 실업수당 청구건수', '미국 소비율']:
        print(key, partition_entropy_by(inputs, key))
    print()

    # senior_inputs = [(input, label)
    #                  for input, label in inputs if input["level"] == "Senior"]
    #
    # for key in ['lang', 'tweets', 'phd']:
    #     print(key, partition_entropy_by(senior_inputs, key))
    # print()

    print("building the tree")
    tree = build_tree_id3(inputs)
    print(tree)

    # print("Intern", classify(tree, { "level" : "Intern" } ))
    # print("Senior", classify(tree, { "level" : "Senior" } ))
