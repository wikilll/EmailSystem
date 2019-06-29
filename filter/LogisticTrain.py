import re
import os
import time
import jieba
import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import liblinear

with open('../data/full/index') as file:
    y = [k.split()[0] for k in file.readlines()]
print(len(y))
#
# with open('./data/full/index') as file:
#     filePath_list = ['./data' + k.strip().split()[1][2:] for k in file.readlines()]


def getFilePathList2(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

startTime = time.time()
filePath_list = getFilePathList2('../data/data')
print(len(filePath_list))
print(filePath_list[0])
print(filePath_list[1])
print('function use %.2f seconds' %(time.time()-startTime))

mailContent_list = []
for filePath in filePath_list:
    with open(filePath, errors='ignore') as file:
        file_str = file.read()
        mailContent = file_str.split('\n\n', maxsplit=1)[1]
        mailContent_list.append(mailContent)
print(mailContent_list[1])

mailContent_list = [re.sub('\s+', ' ', k) for k in mailContent_list]

# with open('stopwords.txt', encoding='utf8') as file:
#     file_str = file.read()
#     stopword_list = file_str.split('\n')
#     stopword_set = set(stopword_list)
# print(len(stopword_list))
# print(len(stopword_set))
#
# cutWords_list = []
# startTime = time.time()
# i = 0
# for mail in mailContent_list[:3000]:
#     cutWords = [k for k in jieba.lcut(mail) if k not in stopword_set]
#     cutWords_list.append(cutWords)
#     i += 1
#     if i % 1000 == 0:
#         print('前%d篇邮件分词共花费%.2f秒' %(i, time.time()-startTime))
#
#
# with open('cutWords_list.pickle', 'wb') as file:
#     pickle.dump(cutWords_list, file)

with open('cutWords_list.pickle', 'rb') as file:
    cutWords_list = pickle.load(file)


tfidf = TfidfVectorizer(cutWords_list, min_df=100, max_df=0.25)

X = tfidf.fit_transform(mailContent_list)
# print('词表大小:', len(tfidf.vocabulary_))
# print(X.shape)
#
# labelEncoder = LabelEncoder()
# y_encode = labelEncoder.fit_transform(y)
#
# train_X, test_X, train_y, test_y = train_test_split(X, y_encode, test_size=0.2)
# logistic_model = LogisticRegressionCV()
# logistic_model.fit(train_X, train_y)
# logistic_model.score(test_X, test_y).round(4)
#
# with open('allModel.pickle', 'wb') as file:
#     save = {
#         'labelEncoder' : labelEncoder,
#         'tfidfVectorizer' : tfidf,
#         'logistic_model' : logistic_model
#     }
#     pickle.dump(save, file)

with open('allModel.pickle', 'rb') as file:
    allModel = pickle.load(file)
    labelEncoder = allModel['labelEncoder']
    tfidfVectorizer = allModel['tfidfVectorizer']
    logistic_model = allModel['logistic_model']

# cv_split = ShuffleSplit(n_splits=5)
# logisticCV_model = LogisticRegressionCV()
# score_ndarray = cross_val_score(logisticCV_model, X, y, cv=cv_split)
# print(score_ndarray)
# print(score_ndarray.mean())

predict_y = logistic_model.predict(X)
#predict_y = list(map(lambda x: str(x), predict_y))
pd.DataFrame(confusion_matrix(y, predict_y),
            columns=labelEncoder.classes_,
            index=labelEncoder.classes_)


def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]

eval_model(y, predict_y, labelEncoder.classes_)