# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from rnn_model import TRNNConfig, TextRNN
from data.cnews_loader import read_category, read_vocab
import ipdb
import pandas as pd
import numpy as np
import sys

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = "./data"
cnn_base_dir = os.path.join(base_dir, 'cnn_data')
rnn_base_dir = os.path.join(base_dir, 'rnn_data')
vocab_dir = os.path.join(cnn_base_dir, 'apptype_vocab.txt')

cnn_save_dir = './checkpoints/textcnn'
rnn_save_dir = './checkpoints/textrnn'
cnn_save_path = os.path.join(cnn_save_dir, 'best_validation')  # 最佳验证结果保存路径
rnn_save_path = os.path.join(rnn_save_dir, 'best_validation')  # 最佳验证结果保存路径

categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories = categories
        self.cat_to_id = cat_to_id
        self.words = words
        self.word_to_id = word_to_id
        self.config.vocab_size = len(self.words)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.session.as_default():
            with self.graph.as_default():
                self.model = TextCNN(self.config)
                self.session.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
                self.saver.restore(sess=self.session, save_path=cnn_save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        with self.session.as_default():
            with self.session.graph.as_default():
                self.cnn_logits = self.session.run(self.model.logits_softmax, feed_dict=feed_dict)
        return self.cnn_logits

class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories = categories
        self.cat_to_id = cat_to_id
        self.words = words
        self.word_to_id = word_to_id
        self.config.vocab_size = len(self.words)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.session.as_default():
            with self.graph.as_default():
                self.model = TextRNN(self.config)
                self.session.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
                self.saver.restore(sess=self.session, save_path=rnn_save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        with self.session.as_default():
            with self.session.graph.as_default():
                self.rnn_logits = self.session.run(self.model.logits_softmax, feed_dict=feed_dict)
        return self.rnn_logits


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['cnn', 'rnn', 'bagging', 'boosting']:
        raise ValueError("""usage: python bagging_predict.py [cnn / rnn / bagging / boosting]""")

    dictionary = os.path.join(cnn_base_dir, "apptype_train_categories.csv")
    dic = pd.read_csv(dictionary, encoding="utf-8")
    dic.drop(["Unnamed: 0"], axis=1, inplace=True)
    dic_backup = dict()
    for index, colmn in dic.iteritems():
        dic_backup.update({colmn.values[0]: index})

    predict_path = os.path.join(base_dir, "app_test_tokinize.csv")
    test_txt = pd.read_csv(predict_path)

    dict = {"id": [], "label1": [], "label2": []}
    result = pd.DataFrame(dict)

    if sys.argv[1] == 'cnn':
        cnn_model = CnnModel()
        for index, rows in test_txt.iterrows():
            i = result.shape[0]
            cnn_predict_result = cnn_model.predict(rows["1"])
            predict_result_1 = categories[cnn_predict_result[0][-1]]
            predict_result_2 = categories[cnn_predict_result[0][-2]]
            result.loc[i + 1] = [rows["0"], predict_result_1, predict_result_2]
            print(index)

    elif sys.argv[1] == 'rnn':
        rnn_model = RnnModel()
        for index, rows in test_txt.iterrows():
            i = result.shape[0]
            rnn_predict_result = rnn_model.predict(rows["1"])
            predict_result_1 = categories[rnn_predict_result[0][-1]]
            predict_result_2 = categories[rnn_predict_result[0][-2]]
            result.loc[i + 1] = [rows["0"], predict_result_1, predict_result_2]
            print(index)

    elif sys.argv[1] == 'bagging':
        cnn_model = CnnModel()
        rnn_model = RnnModel()
        for index, rows in test_txt.iterrows():
            i = result.shape[0]
            cnn_predict_result = cnn_model.predict(rows["1"])
            rnn_predict_result = rnn_model.predict(rows["1"])
            bagging_predict_result = cnn_predict_result + rnn_predict_result
            predict_result_1 = categories[bagging_predict_result[0][-1]]
            predict_result_2 = categories[bagging_predict_result[0][-2]]
            result.loc[i + 1] = [rows["0"], predict_result_1, predict_result_2]
            print(index)

    elif sys.argv[1] == 'boosting':
        print(" Current no")

    result = result.replace(dic_backup)
    result.to_csv(os.path.join(base_dir, "pre_test_9.csv"), index=None, encoding="utf-8")
