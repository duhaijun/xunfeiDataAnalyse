# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab
import ipdb
import pandas as pd

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = './data'
vocab_dir = os.path.join(base_dir, 'apptype_vocab.txt')

save_dir = './checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        # y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        #  return self.categories[y_pred_cls[0]]

        y_pred_cls = self.session.run(self.model.y_pred_cls_2, feed_dict=feed_dict)
        return self.categories[y_pred_cls[1][0][0]], self.categories[y_pred_cls[1][0][1]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    dictionary = os.path.join(base_dir, "apptype_train_categories.csv")
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
        predict_result_1,  predit_result_2 = cnn_model.predict(rows["1"])
        result.loc[i+1] = [rows["0"], predict_result_1, predit_result_2]
        print (index)

    result = result.replace(dic_backup)
    result.to_csv(os.path.join(base_dir, "pre_test_7.csv"), index=None, encoding="utf-8")