# -*- coding: UTF-8 -*-
import pandas as pd
import ipdb
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
import chardet
import jieba
import SynonymsReplacer
import augment

train_path = "/home/lmc/dhj/xunfei/apptype_train.dat"
train_name = "/home/lmc/dhj/xunfei/apptype_id_name.txt"
train_txt = "/home/lmc/dhj/text-classification-cnn-rnn/data/apptype_train.txt"
val_txt = "/home/lmc/dhj/text-classification-cnn-rnn/data/apptype_val.txt"
stop_word_path = "/home/lmc/dhj/text-classification-cnn-rnn/data/stop_word.txt"
same_word = "/home/lmc/dhj/tradtitionalMLToTextClassification/data/sameMeanWord.txt"
train = "/home/lmc/dhj/text-classification-cnn-rnn/data/train.csv"
small_train = "/home/lmc/dhj/text-classification-cnn-rnn/data/small_train.csv"
train_augment = "/home/lmc/dhj/text-classification-cnn-rnn/data/train_augment.csv"
predict_path = "./data/app_test_tokinize.csv"
gen_path = "/home/lmc/dhj/tensorflow_poems/data/gen.csv"

train_df = pd.read_csv("/home/lmc/dhj/text-classification-cnn-rnn/data/train_end.csv", encoding="utf-8")

# train_augment_df = pd.read_csv(train_augment, encoding="utf-8")
# gen_df = pd.read_csv(gen_path, encoding="utf-8")
# gen_df["2"] = "gen"
#
# zero_df = gen_df["0"] # 140901
# one_df = gen_df["1"] # 工具
# two_df = gen_df["2"] # gen
# gen_df.drop(["0", "1", "2"], axis=1, inplace=True)
# gen_df["0"] = two_df
# gen_df["1"] = zero_df
# gen_df["2"] = one_df
# train_df = pd.concat([train_augment_df, gen_df], axis=0)


# ipdb.set_trace()
#
# train_df = train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[20:])]
# train_df = train_df.sample(frac=1)
# train_df.drop(["0"], axis=1, inplace=True)
# ipdb.set_trace()
#
# train_df.to_csv("/home/lmc/dhj/tensorflow_poems/data/train.txt", index=None, header=None, sep=":", encoding="utf-8")
#
# ipdb.set_trace()

# stop_word = list()
# with open(stop_word_path, "r") as f:
#     for line in f:
#         stop_word.append(line.strip())

# train_df = pd.read_table(train_path, sep="\\t", header=None, encoding="utf-8")
# for i, rows in train_df.iterrows():
#     word = jieba.cut(rows[2])
#     word_list = [val for val in word if val not in stop_word]
#     train_df[2].iloc[i] = word_list
#
#     if "|" in rows[1]:
#         left = rows[1][:rows[1].index("|")]
#         right = rows[1][rows[1].index("|")+1:]
#         train_df.loc[i][1] = left
#         train_df.loc[train_df.index.max()+1] = [train_df.loc[i][0], right, train_df.loc[i][2]]

# train_df = train_df.reset_index(drop=True)
# train_df.to_csv(train, encoding="utf-8")
#
# train_df = pd.read_csv(train, encoding="utf-8")
# train_df["1"] = train_df[1].astype(str)
# train_df.drop(["Unnamed: 0"], axis=1, inplace=True)
# train_df.to_csv(train, encoding="utf-8", index=None)
#
# train_df.columns = ["0", "1", "2"]


# test_txt = pd.read_csv(predict_path)
# #
# ipdb.set_trace()
# test_txt.drop(["Unnamed: 0"], axis=1, inplace=True)
# test_txt.columns = ["0", "1"]
# #
# for i, rows in test_txt.iterrows():
#     word = jieba.cut(rows["1"])
#     word_list = [val for val in word if val not in stop_word]
#     test_txt["1"].iloc[i] = word_list
# #
# ipdb.set_trace()
# test_txt.to_csv(predict_path, encoding="utf-8", index=None)
# ipdb.set_trace()

# train_df = pd.read_csv(train_augment, encoding="utf-8")
#
# j = 0
# for i in range(6):
#     for k in range(i):
#         if i == 5:
#             train_df = pd.concat([train_df, train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[j*10:])]], axis=0)
#         else: train_df = pd.concat([train_df, train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[j*10:(j+1)*10])]], axis=0)
#     j+=1
# # 少样本数据
# big_df_one = train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[:1])].sample(frac=0.2)
# big_df_two = train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[1:2])].sample(frac=0.5)

# big_df = pd.concat([big_df_one, big_df_two], axis=0)
# small_df = train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[2:])]
#
# #
# train_df = pd.concat([big_df, small_df], axis=0)
# train_df.to_csv(train_augment, encoding="utf-8", index=None)

# train_df = pd.read_csv(train_augment, encoding="utf-8")
train_df["1"] = train_df["1"].astype(str)

big_data_one = train_df[train_df["1"].isin([train_df["1"].value_counts().index.tolist()[0]])]
big_data_two = train_df[train_df["1"].isin([train_df["1"].value_counts().index.tolist()[1]])]
other_data = train_df[train_df["1"].isin(train_df["1"].value_counts().index.tolist()[2:])]

# 这里的train_data是cnn的训练数据
train_data = pd.concat([big_data_one.iloc[:800], other_data], axis=0)
train_data = pd.concat([train_data, big_data_two[:600]], axis=0)
train_data = train_data.sample(frac=1, replace=True)
# 这里的val_data是rnn的训练数据
val_data = pd.concat([big_data_one[800:], other_data], axis=0)
val_data = pd.concat([val_data, big_data_two[600:]], axis=0)
val_data = val_data.sample(frac=1, replace=True)



# 数据写为txt格式，适用于text-classification-cnn-rnn
dic = dict()
all_dic = dict()
with open(train_name,"rb") as d:
    for line in d:
        nameId, name = line.strip().split("\t")
        all_dic[nameId] = name

for i, Id in train_data.iterrows():
    if Id["1"] not in dic.keys():
        dic[Id["1"]] = all_dic[Id["1"]]

df = pd.DataFrame([dic])
df.to_csv("/home/lmc/dhj/text-classification-cnn-rnn/data/apptype_train_categories.csv")

with open(train_txt, "wb") as f:
    for index, sentence in train_data.iterrows():
        f.write(dic[sentence["1"]] + "\t" + sentence["2"]+"\n")

with open(val_txt, "wb") as f:
    for _, sentence in val_data.iterrows():
        f.write(dic[sentence["1"]] + "\t" + sentence["2"]+"\n")



