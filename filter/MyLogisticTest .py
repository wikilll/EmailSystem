# coding=utf8
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba
import os

def getFilePathList2(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

filePath_list = getFilePathList2('../data/data')
print(len(filePath_list))

mailContent_list = []
for filePath in filePath_list:
    with open(filePath, errors='ignore') as file:
        file_str = file.read()
        mailContent = file_str.split('\n\n', maxsplit=1)[1]
        mailContent_list.append(mailContent)

mailContent_list = [re.sub('\s+', ' ', k) for k in mailContent_list]

myMailContent = "汽车工业是我国重要的支柱产业，该产业具有高入、高产出、高效益的特点，是典型的资金技术密型产业,对刺激我国经济增长有很大作用。然而汽车工业的发展也带来了不少问题，如资源紧缺、环境污染、交通拥挤及事故频发等亟需解决的问题。下面，我将从多个角度提出对我国汽车工业发展的建议。"
myMailContent = re.sub('\s+', ' ', myMailContent)
mailContent_list.append(myMailContent)
print(myMailContent)

with open('stopwords.txt', encoding='utf8') as file:
    file_str = file.read()
    stopword_list = file_str.split('\n')
    stopword_set = set(stopword_list)
print(len(stopword_list))
print(len(stopword_set))

cutWords_list = []
i = 0
for mail in mailContent_list[:3000]:
    cutWords = [k for k in jieba.lcut(mail) if k not in stopword_set]
    cutWords_list.append(cutWords)
    i += 1

tfidf = TfidfVectorizer(cutWords_list, min_df=100, max_df=0.25)

X = tfidf.fit_transform(mailContent_list)
print(len(tfidf.vocabulary_))
print(X.shape)

with open('allModel.pickle', 'rb') as file:
    allModel = pickle.load(file)
    logistic_model = allModel['logistic_model']

predict_y = logistic_model.predict(X)

print(predict_y[-1])
