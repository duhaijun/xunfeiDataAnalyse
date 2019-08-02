## 本项目是2019讯飞文本分类竞赛的部分参赛代码，包括TextCNN和TextRNN部分。
##### 代码基于 https://github.com/gaussic/text-classification-cnn-rnn 做了部分修改。

# 数据分析
<p align="center">
	<img src="https://www.showdoc.cc/server/api/common/visitfile/sign/eec56ce82c8bddef3afd1bdfc481bdab?showdoc=.jpg" alt="Sample"  width="800" height="300">
	图1 训练数据类别数统计直方图
</p>

<p align="center">
	<img src="https://www.showdoc.cc/server/api/common/visitfile/sign/f830fcc1cc732f500e1f84046640ae8f?showdoc=.jpg" alt="Sample"  width="600" height="600">
	图2 训练数据类别数统计饼图
</p>

从上图1和图2中可以看出，训练数据存在严重的类别不平衡问题，有两个类的数量远远超过其他类，有的类的数据量非常少。因此，后续第一步要解决的就是训练数据类别不平衡问题。
同时，总的数据量不是很大，可以考虑用SVM等方法。

<p align="center">
	<img src="https://www.showdoc.cc/server/api/common/visitfile/sign/0df210cadf10987d68d51eebd5201d28?showdoc=.jpg" alt="Sample"  width="600" height="600">
	图3 字符长长度统计结果饼图
</p>

<p align="center">
	<img src="https://www.showdoc.cc/server/api/common/visitfile/sign/e3a9acb16837056192b04efb10e3e4fa?showdoc=.jpg" alt="Sample"  width="700" height="300">
	图4 字符串长度统计直方图
</p>

从图3和图4中可以看出，训练数据中的字符串长度大部分都小于2000，均值为600。因此，后续在模型中，最大字符串长度这个参数可以取均值的两倍。


# 数据处理和模型训练
##### 基于以上的数据分析结果，对数据进行以下处理：
1. 首先对训练和测试数据中的文本进行了jieba分词，去停词处理。删除无意义的词和停词表中的词。（其中停词表是从博客复制来的：https://blog.csdn.net/dorisi_h_n_q/article/details/82114913）

2. 然后采用TextCNN的深度神经网络跑了一下，得分为 52.69353。（TextCNN地址：https://github.com/gaussic/text-classification-cnn-rnn）
其中随机选取了训练数据的80%作为训练集，剩下20%作为验证集。（没有使用交叉验证）

3. 使用诗歌生成模型做数据增强，对数量较少的类别进行数据扩增。（诗歌生成模型地址：https://github.com/jinfagang/tensorflow_poems）
数据扩增中，选取了类别数量top20之后的所有数据，并将数据量统一补齐到300。训练中，类别名作为输入，app描述作为label。

4. 对数量最多的两个类别做欠采样，缓解数据不平衡问题。（最多的一个**[140901]**取0.3，第二多的**[140206]**取0.5）

5. 采用TextCNN跑了一下，得分为 64.7895

6. 用bert取代TextCNN跑一下，得分为 76.07062

# 模型融合
在上述部分中，BERT模型得到的结果已经是较好的了，后续提升的幅度不大。因此，考虑通过模型融合对结果进行进一步的提升。

思路：先把TextCNN和TextRNN进行融合，在将融合后的模型与BERT进行融合。

项目中bagging_predict.py对应的是TextCNN和TextRNN使用bagging方法进行融合；
![bagging](https://www.showdoc.cc/server/api/common/visitfile/sign/2b0a9b123de902186a39c4a7b0fcc909?showdoc=.jpg "bagging")

boosting_data_processing.py实现的功能是在TextCNN训练完后，对训练集样本的权重进行更新并生成更新后的训练集给TextRNN用。
