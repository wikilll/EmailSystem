# -*- coding:utf-8 -*-

import time

import joblib
import pandas as pd
import matplotlib as mpl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD  # 降维
from sklearn.naive_bayes import BernoulliNB  # 伯努利分布的贝叶斯公式
from sklearn.metrics import f1_score, precision_score, recall_score

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 文件数据读取（分词过后的文件）
df = pd.read_csv("../data/result_process02", encoding="utf-8", sep=",")
# 如果有NaN值，进行删除操作
df.dropna(axis=0, how="any", inplace=True)  # 删除表中含有任何NaN的行

# 数据分割(分成训练数据集和测试数据集）
x_train, x_test, y_train, y_test = train_test_split(df[["has_date", "jieba_cut_content", "content_sema"]],
                                                    df["label"], test_size=0.2, random_state=0)  # x:特征列 y:label列

# 使用贝叶斯算法开始模型训练
# 特征工程，将文本数据转换为数值型数据
transformer = TfidfVectorizer(norm="l2", use_idf=True)  # 获得矩阵 体现字词的重要性
svd = TruncatedSVD(n_components=20)  # 奇异值分解，降维（简化数据）
jieba_cut_content = list(x_train["jieba_cut_content"].astype("str"))
transformer_model = transformer.fit(jieba_cut_content)
df1 = transformer_model.transform(jieba_cut_content)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)
data = pd.DataFrame(df2)

data["has_date"] = list(x_train["has_date"])
data["content_sema"] = list(x_train["content_sema"])

t1 = time.time()
nb = BernoulliNB(alpha=1.0, binarize=0.0005)  # 贝叶斯分类模型构建
model = nb.fit(data, y_train)
t = time.time() - t1
print("贝叶斯模型构建时间为:%.5f ms" % (t * 1000))

# 对测试集数据进行转换
jieba_cut_content_test = list(x_test["jieba_cut_content"].astype("str"))
data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test["has_date"] = list(x_test["has_date"])
data_test["content_sema"] = list(x_test["content_sema"])

# 对测试数据进行测试
y_predict = model.predict(data_test)

# 效果评估
print("准确率为:%.5f" % precision_score(y_test, y_predict))  # P
print("召回率为:%.5f" % recall_score(y_test, y_predict))  # 垃圾邮件的拦截率  R
print("F1值为:%.5f" % f1_score(y_test, y_predict))  # 2*P*R/(P+R)

joblib.dump(model, 'test.pkl')
