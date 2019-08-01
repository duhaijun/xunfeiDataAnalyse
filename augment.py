# -*- coding: utf-8 -*-
#数据增强，shuffle或drop
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import ipdb

# train = "/home/lmc/dhj/text-classification-cnn-rnn/data/train.csv"
#
# train_df = pd.read_csv(train, encoding="utf-8")
# #train_df["question_text"] = train_df["question_text"].map(lambda x: clean_text(x))
# xtrain = train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[20:])]

def shuffle(d):
    return np.random.permutation(d)

def shuffle2(d):
    len_ = len(d)
    times = 2
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]],d[index[1]] = d[index[1]],d[index[0]]
    return d

def dropout(d, p=0.4):
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d

def clean(xx):
    xx2 = re.sub(r'\?', "", xx)
    xx1= xx2.split(' ')
    return xx1

def dataaugment(X):
    for i, rows in X.iterrows():
        item = clean(rows["2"])
        d1 = shuffle2(item)
        d11 = ' '.join(d1)
        d2 = dropout(item)
        d22 = ' '.join(d2)
        X.loc[X.index.max() + 1] = [X["0"].loc[i], X["1"].loc[i], d11]
        X.loc[X.index.max() + 1] = [X["0"].loc[i], X["1"].loc[i], d22]
        print (i)
    return X
