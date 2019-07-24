# xunfeiDataAnalyse
xunfei AI textClassification
# 数据处理
1. 首先对训练和测试数据中的文本进行了jieba分词，去停词处理。
2. 然后采用TextCNN的深度神经网络跑了一下，得分为 52.69353
3. 使用诗歌生成模型做数据增强，对数量较少的类别进行数据扩增。
4. 对数量最多的两个类别做欠采样，缓解数据不平衡问题。
5. 采用TextCNN跑了一下，得分为 64.7895
