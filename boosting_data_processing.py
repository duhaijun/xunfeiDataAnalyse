# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from rnn_model import TRNNConfig, TextRNN
from data.cnews_loader import read_category, read_vocab, process_file
import ipdb
import pandas as pd
import numpy as np
import math


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


def get_model_weight(model, data_name="cnn_data"):
    """
    :param model: categorier
    :param data_name: "cnn_data" or "rnn_data"
    :return: train data after boosting
    """

    train_dir = os.path.join(base_dir, data_name, 'apptype_train.txt')
    dictionary = os.path.join(cnn_base_dir, "apptype_train_categories.csv")

    # 生成字典
    dic = pd.read_csv(dictionary, encoding="utf-8")
    dic.drop(["Unnamed: 0"], axis=1, inplace=True)
    dic_backup = dict()
    for index, colmn in dic.iteritems():
        dic_backup.update({colmn.values[0]: index})

    test_txt = pd.DataFrame(train_dir)

    # data_weight是样本权重
    dict = {"0": [], "1": [], "2":[], "data_weight":[]}
    result = pd.DataFrame(dict)

    categories_true = 0
    categories_false = 0
    for index, rows in test_txt.iterrows():
        i = result.shape[0]
        predict_result = cnn_model.predict(rows["2"])
        predict_result = np.argsort(predict_result)

        # 模型分类结果
        predict_result_1 = categories(predict_result[-1])
        if dic_backup[predict_result_1] == rows["1"]:
            # 如果分类正确，则data_weight=True
            result.loc[i + 1] = [rows["0"], rows["1"], rows["2"], "true"]
            categories_true+=1
        else:
            # 分类错误， data_weight=False
            result.loc[i + 1] = [rows["0"], rows["1"], rows["2"], "false"]
            categories_false+=1

        print (index)

    # 根据博客https://blog.csdn.net/csqazwsxedc/article/details/72354206给的公式计算
    errorStatistics = float(categories_false)/float(categories_false+categories_true)
    aerfa = 0.5*math.log((1-errorStatistics)/errorStatistics)
    weight_dic = {"true":1*math.exp(-aerfa)/float(categories_true+categories_false),
                  "false":1*math.exp(aerfa)/float(categories_true+categories_false)}

    # 通过字典，把分类正确的样本权重变化值放到data_weight中
    result = result.replace(weight_dic)
    # 最终得到训练集经过第一个分类器后，样本权重变换后的值组成的第二个分类器的训练集
    result.to_csv(os.path.join(base_dir, data_name, "After_boosting_train_data.csv"), index=None, encoding="utf-8")


if __name__ == '__main__':
    cnn_model = CnnModel()
    rnn_model = RnnModel()

    # 这里以TextCnn为第一个分类器，TextRnn为第二个分类器
    get_model_weight(cnn_model, "cnn_data")