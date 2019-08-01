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

sess = tf.Session()
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
        self.model = TextCNN(self.config)

        self.session = sess
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=cnn_save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

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
        self.model = TextRNN(self.config)

        self.session = sess
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=rnn_save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        self.rnn_logits = self.session.run(self.model.logits_softmax, feed_dict=feed_dict)

        return self.rnn_logits


if __name__ == '__main__':
    cnn_model = CnnModel()
    rnn_model = RnnModel()
    dictionary = os.path.join(cnn_base_dir, "apptype_train_categories.csv")
    dic = pd.read_csv(dictionary, encoding="utf-8")
    dic.drop(["Unnamed: 0"], axis=1, inplace=True)
    dic_backup = dict()
    for index, colmn in dic.iteritems():
        dic_backup.update({colmn.values[0]: index})

    predict_path = os.path.join(base_dir, "app_test_tokinize.csv")
    test_txt = pd.read_csv(predict_path)

    dict = {"id":[], "label1":[], "label2":[]}
    result = pd.DataFrame(dict)

    for index, rows in test_txt.iterrows():
        i = result.shape[0]
        cnn_predict_result = cnn_model.predict(rows["1"])
        rnn_predict_result = rnn_model.predict(rows["1"])
        predict_result = np.argsort(cnn_predict_result + rnn_predict_result)

        ipdb.set_trace()

        predict_result_1 = categories(predict_result[-1])
        predict_result_2 = categories(predict_result[-2])

        result.loc[i+1] = [rows["0"], predict_result_1, predict_result_2]
        print (index)

    result = result.replace(dic_backup)
    result.to_csv(os.path.join(base_dir, "pre_test_8.csv"), index=None, encoding="utf-8")