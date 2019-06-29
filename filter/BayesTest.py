#coding:utf-8
import re

import jieba
from sklearn.externals import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD  #降维
from sklearn.naive_bayes import BernoulliNB     #伯努利分布的贝叶斯公式
from sklearn.metrics import f1_score,precision_score,recall_score

#读取邮件内容
from filter.featureSelector import precess_content_sema


def read_file(file_path):
    # 邮件数据编码为"gb2312",数据读取有异常就ignore
    file = open(file_path,"r",encoding="gb2312",errors="ignore")
    content_dict = {}
    try:
        is_content = False
        for line in file:
            line = line.strip()
            if line.startswith("From:"):
                content_dict["from"] = line[5:]
            elif line.startswith("To:"):
                content_dict["to"] = line[3:]
            elif line.startswith("Date:"):
                content_dict["date"] = line[5:]
            elif not line:
                is_content = True

            if is_content:
                if "content" in content_dict:
                    content_dict["content"] += line
                else:
                    content_dict["content"] = line
    finally:
        file.close()
    return content_dict

#邮件数据处理(内容的拼接,并用逗号进行分割)
def process_file(file_path):
    content_dict = read_file(file_path)

    result_str = content_dict.get("from","unkown").replace(",","").strip()+","
    result_str += content_dict.get("to","unkown").replace(",","").strip()+","
    result_str += content_dict.get("data","unkown").replace(",","").strip()+","
    result_str += content_dict.get("content","unkown").replace(",","").strip()
    return result_str


with open("../data/result", "w", encoding='utf-8') as writer:
    result_str = process_file("000")
    writer.writelines(result_str)
# result_str = process_file("myTest")
df = pd.read_csv("../data/result",sep=",",header=None,
                 names=["from","to","date","content"])

def extract_email_date(str1):
    if not isinstance(str1,str):  #判断变量是否是str类型
        str1 = str(str1)    #str类型的强转
    str_len = len(str1)

#对不同格式的时间信息进行处理
    if str_len < 10:
        week = "unknown"
        hour = "unknown"
        time_quantum ="unknown" # 0表示：上午[8,12]；1表示：下午[13,18]；2表示：晚上[19,23]；3表示：凌晨[0,7]
        pass
    elif str_len == 16:
        #2005-9-2 上午10:55
        rex = r"(\d{2}):\d{2}"  # \d  匹配任意数字
        it = re.findall(rex,str1)
        if len(it) == 1:
            hour = it[0]
        else:
            hour = "unknown"
        week = "Fri"
        time_quantum = "0"
        pass
    elif str_len == 19:
        #Sep 23 2005 1:04 AM
        week = "Sep"
        hour = "01"
        time_quantum = "3"
        pass
    elif str_len == 21:
        #August 24 2005 5:00pm
        week = "Wed"
        hour = "17"
        time_quantum = "1"
        pass
    else:
        #匹配一个字符开头，+表示至少一次  \d 表示数字   ？表示可有可无  *? 非贪婪模式
        rex = r"([A-Za-z]+\d?[A-Za-z]*) .*?(\d{2}):\d{2}:\d{2}.*"
        it = re.findall(rex,str1)
        if len(it) == 1 and len(it[0]) == 2:
            week = it[0][0][-3]
            hour = it[0][1]
            int_hour = int(hour)
            if int_hour < 8:
                time_quantum = "3"
            elif int_hour < 13:
                time_quantum = "0"
            elif int_hour < 19:
                time_quantum = "1"
            else:
                time_quantum = "2"
            pass
        else:
            week = "unknown"
            hour = "unknown"
            time_quantum = "unknown"
    week = week.lower()
    hour = hour.lower()
    time_quantum = time_quantum.lower()
    return (week,hour,
            time_quantum)

#数据转换
data_time_extract_result = list(map(lambda st:extract_email_date(st),df["date"]))
df["date_week"] = pd.Series(map(lambda t:t[0],data_time_extract_result))
df["date_hour"] = pd.Series(map(lambda t:t[1],data_time_extract_result))
df["date_time_quantum"] = pd.Series(map(lambda t:t[2],data_time_extract_result))
df["has_date"] = df.apply(lambda c: 0 if c["date_week"] == "unknown" else 1,axis=1)
df["jieba_cut_content"] = list(map(lambda st:" ".join(jieba.cut(st)),df["content"]))
df["content_length"] = pd.Series(map(lambda st:len(st),df["content"]))
df["content_sema"] = list(map(lambda st:precess_content_sema(st),df["content_length"]))


transformer = TfidfVectorizer(norm="l2",use_idf=True)     #词频-逆向文件频率。在处理文本时，如何将文字转化为模型可以处理的向量呢？IF-IDF就是这个问题的解决方案之一。字词的重要性与其在文本中出现的频率成正比(IF)，与其在语料库中出现的频率成反比(IDF)。
svd = TruncatedSVD(n_components=20)     #奇异值分解，降维（简化数据）
jieba_cut_content_test = list(df["jieba_cut_content"].astype("str"))
transformer_model = transformer.fit(jieba_cut_content_test)
df1 = transformer_model.transform(jieba_cut_content_test)
svd_model = svd.fit(df1)
df2 = svd_model.transform(df1)

data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(jieba_cut_content_test)))
data_test["has_date"] = list(df["has_date"])
data_test["content_sema"] = list(df["content_sema"])

model = joblib.load('test.pkl')

a = model.predict(data_test)
print(a)