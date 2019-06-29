# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import jieba

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 文件数据读取
df = pd.read_csv("../data/result_process01", sep=",", header=None,
                 names=["from", "to", "date", "content", "label"])


# 邮件的时间信息提取
def extract_email_date(str1):
    if not isinstance(str1, str):  # 判断变量是否是str类型
        str1 = str(str1)  # str类型的强转
    str_len = len(str1)

    # week:星期
    # hour:小时
    # time_quantum:时间段

    # 对不同格式的时间信息进行处理
    if str_len < 10:
        week = "unknown"
        hour = "unknown"
        time_quantum = "unknown"  # 0表示：上午[8,12]；1表示：下午[13,18]；2表示：晚上[19,23]；3表示：凌晨[0,7]
        pass
    elif str_len == 16:
        # 2005-9-2 上午10:55
        rex = r"(\d{2}):\d{2}"  # \d  匹配任意数字
        it = re.findall(rex, str1)
        if len(it) == 1:
            hour = it[0]
        else:
            hour = "unknown"
        week = "Fri"
        time_quantum = "0"
        pass
    elif str_len == 19:
        # Sep 23 2005 1:04 AM
        week = "Sep"
        hour = "01"
        time_quantum = "3"
        pass
    elif str_len == 21:
        # August 24 2005 5:00pm
        week = "Wed"
        hour = "17"
        time_quantum = "1"
        pass
    else:
        # 匹配一个字符开头，+表示至少一次  \d 表示数字   ？表示可有可无  *? 非贪婪模式
        rex = r"([A-Za-z]+\d?[A-Za-z]*) .*?(\d{2}):\d{2}:\d{2}.*"
        it = re.findall(rex, str1)
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
    return (week, hour, time_quantum)


# 数据转换
data_time_extract_result = list(map(lambda st: extract_email_date(st), df["date"]))
df["date_week"] = pd.Series(map(lambda t: t[0], data_time_extract_result))
df["date_hour"] = pd.Series(map(lambda t: t[1], data_time_extract_result))
df["date_time_quantum"] = pd.Series(map(lambda t: t[2], data_time_extract_result))

# 添加是否有时间（没有时间肯定是垃圾邮件）
# 其他的属性对垃圾邮件的分类作用不大
df["has_date"] = df.apply(lambda c: 0 if c["date_week"] == "unknown" else 1, axis=1)

# jieba分词
# 为算法的输入（特征矩阵）做准备
df["content"] = df["content"].astype("str")
df["jieba_cut_content"] = list(map(lambda st: " ".join(jieba.cut(st)), df["content"]))  # 分开的词用空格隔开

# 邮件信息量（长度）对垃圾邮件识别的影响
df["content_length"] = pd.Series(map(lambda st: len(st), df["content"]))

# 添加信号量
# 文本长度:x 调节因子:500&10000 信息量平滑因子:1&1
def precess_content_sema(x):
    if x > 10000:
        return 0.5 / np.exp(np.log10(x) - np.log10(500)) + np.log(abs(x - 500) + 1) - np.log(abs(x - 10000)) + 1
    else:
        return 0.5 / np.exp(np.log10(x) - np.log10(500)) + np.log(abs(x - 500) + 1) + 1


df["content_sema"] = list(map(lambda st: precess_content_sema(st), df["content_length"]))

# 获取需要的列,删除不需要的列
df.drop(["from", "to", "date", "content",
         "date_week", "date_hour", "date_time_quantum", "content_length"], 1, inplace=True)

# 结果输出到CSV文件中
df.to_csv("../data/result_process02", encoding="utf-8", index=False)
