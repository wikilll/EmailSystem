#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import jieba
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)


#设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#文件数据读取
df = pd.read_csv("../data/result_process01",sep=",",header=None,
                 names=["from","to","date","content","label"])

#特征工程一：邮件服务器
#提取发件人和收件人的邮件服务器地址
def extract_email_server_address(str1):
    it = re.findall(r"@([A-Za-z0-9]*\.[A-Za-z0-9\.]+)",str(str1))
    result = ""
    if len(it)>0:
        result = it[0]
    if not result:
        result = "unknown"
    return result

df["to_address"] = pd.Series(map(lambda str:extract_email_server_address(str),df["to"]))
df["from_address"] = pd.Series(map(lambda str:extract_email_server_address(str),df["from"]))

#查看邮件服务器的数量
print("=================to address================")
print(df.to_address.value_counts().head(5))
print("总邮件接收服务器类别数量为:"+str(df.to_address.unique().shape))

print("=================from address================")
print(df.from_address.value_counts().head(5))
print("总邮件接收服务器类别数量为:"+str(df.from_address.unique().shape))

from_address_df = df.from_address.value_counts().to_frame()
len_less_10_from_adderss_count = from_address_df[from_address_df.from_address<=10].shape
print("发送邮件数量小于10封的服务器数量为:"+str(len_less_10_from_adderss_count))

#特征工程二：
#邮件的时间信息提取
def extract_email_date(str1):
    if not isinstance(str1,str):  #判断变量是否是str类型
        str1 = str(str1)    #str类型的强转
    str_len = len(str1)

    week = ""#星期
    hour = ""#小时
    time_quantum = ""#时间段

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
    return (week,hour,time_quantum)

#数据转换
data_time_extract_result = list(map(lambda st:extract_email_date(st),df["date"]))
df["date_week"] = pd.Series(map(lambda t:t[0],data_time_extract_result))
df["date_hour"] = pd.Series(map(lambda t:t[1],data_time_extract_result))
df["date_time_quantum"] = pd.Series(map(lambda t:t[2],data_time_extract_result))
print(df.head(2))

print("=======星期属性字段描述======")
print(df.date_week.value_counts().head(3))
print(df[["date_week","label"]].groupby(["date_week","label"])["label"].count())

print("=======小时属性字段描述======")
print(df.date_hour.value_counts().head(3))
print(df[["date_hour", "label"]].groupby(['date_hour', 'label'])['label'].count())

print("=======时间段属性字段描述======")
print(df.date_hour.value_counts().head(3))
print(df[["date_time_quantum","label"]].groupby(["date_time_quantum","label"])["label"].count())

#添加是否有时间（没有时间肯定是垃圾邮件）
#其他的属性对垃圾邮件的分类作用不大
df["has_date"] = df.apply(lambda c: 0 if c["date_week"] == "unknown" else 1,axis=1)
print(df.head(2))

#特征工程三：
#jieba分词
#为算法的输入（特征矩阵）做准备
df["content"] = df["content"].astype("str")

#jieba添加分词字典，字典格式为：单词 词频(可选的) 词性(可选的)
#词典构建方式：一般都是基于jieba分词之后的效果进行人工干预
#jieba.cut: def cut(self, sentence, cut_all=False, HMM=True)
#   sentence:需要分割的文本，cut_all:分割模式，分为精准模式False、全分割True，HMM：新词可进行推测
#长文本采用精准分割，短文本采用全分割模式
#一般在短文本处理过程中还需要考虑词性，并且还可能将分割好的单词进行组合
df["jieba_cut_content"] = list(map(lambda st:" ".join(jieba.cut(st)),df["content"]))    #分开的词用空格隔开
print(df.head(2))

#特征工程四：
# 邮件信息量（长度）对垃圾邮件识别的影响
def process_content_length(lg):
    if lg < 10:
        return 0
    elif lg <= 100:
        return 1
    elif lg <= 500:
        return 2
    elif lg <= 1000:
        return 3
    elif lg <= 1500:
        return 4
    elif lg <= 2000:
        return 5
    elif lg <= 2500:
        return 6
    elif lg <= 3000:
        return 7
    elif lg <= 4000:
        return 8
    elif lg <= 5000:
        return 9
    elif lg <= 10000:
        return 10
    elif lg <= 20000:
        return 11
    elif lg <= 30000:
        return 12
    elif lg <= 50000:
        return 13
    else:
        return 14

df["content_length"] = pd.Series(map(lambda st:len(st),df["content"]))
df["content_length_type"] = pd.Series(map(lambda st:process_content_length(st),df["content_length"]))
#按照邮件长度类别和标签进行分组groupby，抽取这两列数据相同的放到一起
# 用agg和内置函数count聚合不同长度邮件分贝是否为垃圾邮件的数量
# reset_insex:将对象重新进行索引的构建
df2 = df.groupby(["content_length_type","label"])["label"].agg(["count"]).reset_index()
#label == 1：是垃圾邮件，对长度和数量进行重命名，count命名为c1
df3 = df2[df2.label == 1][["content_length_type","count"]].rename(columns={"count":"c1"})
df4 = df2[df2.label == 0][["content_length_type","count"]].rename(columns={"count":"c2"})
df5 = pd.merge(df3,df4)  #数据集的合并，pandas.merge可依据一个或多个键将不同DataFrame中的行连接起来

df5["c1_rage"] = df5.apply(lambda r:r["c1"]/(r["c1"]+r["c2"]),axis=1)   #按行进行统计
df5["c2_rage"] = df5.apply(lambda r:r["c2"]/(r["c1"]+r["c2"]),axis=1)
print(df5.head())

#画图
plt.plot(df5["content_length_type"],df5["c1_rage"],label=u"垃圾邮件比例")
plt.plot(df5["content_length_type"],df5["c2_rage"],label=u"正常邮件比例")
plt.xlabel(u"邮件长度标记")
plt.ylabel(u"邮件比例")
plt.grid(True)
plt.legend(loc=0)
plt.savefig("垃圾和正常邮件比例.png")
plt.show()

#特征工程五：
#  添加信号量
#文本长度 调节因子 信息量平滑因子
def precess_content_sema(x):
    if x>10000:
        return 0.5/np.exp(np.log10(x)-np.log10(500))+np.log(abs(x-500)+1)-np.log(abs(x-10000))+1
    else:
        return 0.5/np.exp(np.log10(x)-np.log10(500))+np.log(abs(x-500)+1)+1

a = np.arange(1,20000)
plt.plot(a,list(map(lambda t:precess_content_sema(t),a)),label=u"信息量")
plt.grid(True)
plt.legend(loc=0)
plt.savefig("信息量.png")
plt.show()

df["content_sema"] = list(map(lambda st:precess_content_sema(st),df["content_length"]))
print(df.head(2))
#查看列名称
print(df.dtypes)

#获取需要的列,drop删除不需要的列
df.drop(["from","to","date","content","to_address","from_address",
         "date_week","date_hour","date_time_quantum","content_length",
         "content_length_type"],1,inplace=True)
print(df.info())
print(df.head())

#结果输出到CSV文件中
df.to_csv("../data/result_process02",encoding="utf-8",index=False)

