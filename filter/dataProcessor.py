#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

#读取文件路径
def read_index_file(file_path):
    type_dict = {"spam":"1","ham":"0"}      #邮件分类标签
    index_file = open(file_path)
    index_dict = {} #存放邮件路径及其标识
    try:
        for line in index_file:  # 按行读取文件
            arr = line.split(" ")  # 用空格进行分割
            if len(arr) == 2:
                key,value = arr     # 如果长度为2 则将对应的内容赋值给键值对
            value = value.replace("../data","").replace("\n","")    #将路径进行替换
            index_dict[value] = type_dict[key.lower()]
    finally:
        index_file.close()
    return index_dict

#读取邮件内容
def read_file(file_path):
    # 邮件数据编码为"gb2312",数据读取有异常就ignore
    file = open(file_path,"r",encoding="gb2312",errors="ignore")
    content_dict = {}

    try:
        is_content = False
        for line in file:  # 按行读取
            line = line.strip()  # 去掉空格
            #对每一行进行不同的识别处理
            if line.startswith("From:"):
                content_dict["from"] = line[5:]
            elif line.startswith("To:"):
                content_dict["to"] = line[3:]
            elif line.startswith("Date:"):
                content_dict["date"] = line[5:]
            elif not line:
                # 邮件内容与上面信息存在着第一个空行，遇到空行时，这里标记为True以便进行下面的邮件内容处理
                is_content = True

            # 处理邮件内容（处理到为空的行时接着处理邮件的内容）
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

    #进行邮件内容的拼接,get()函数返回指定键的值，指定键的值不存在用指定的默认值unkown代替
    result_str = content_dict.get("from","unkown").replace(",","").strip()+","
    result_str += content_dict.get("to","unkown").replace(",","").strip()+","
    result_str += content_dict.get("data","unkown").replace(",","").strip()+","
    result_str += content_dict.get("content","unkown").replace(",","").strip()
    return result_str

#开始进行数据处理
## os.listdir    返回指定的文件夹包含的文件或文件夹包含的名称列表
index_dict = read_index_file('../data/full/index')
list0 = os.listdir('../data/data')      #list0是范围为[000-215]的列表
for l1 in list0:    # l1:循环000--215
    l1_path = '../data/data/' + l1      #l1_path   ../data/data/215
    print('开始处理文件夹:' + l1_path)
    list1 = os.listdir(l1_path)     #list1:['000', '001', '002', '003'....'299']
    write_file_path = '../data/process01_' + l1
    with open(write_file_path, "w", encoding='utf-8') as writer:
        for l2 in list1:  # l2:循环000--299
            l2_path = l1_path + "/" + l2  # l2_path   ../data/data/215/000
            # 得到具体的文件内容后，进行文件数据的读取
            index_key = "/" + l1 + "/" + l2  # index_key:  /215/000

            if index_key in index_dict:
                # 读取数据
                content_str = process_file(l2_path)
                # 添加分类标签（0、1）用逗号隔开
                content_str += "," + index_dict[index_key] + "\n"
                # 进行数据输出
                writer.writelines(content_str)

# 合并所有第一次构建好的内容
with open('../data/result_process01', 'w', encoding='utf-8') as writer:
    for l1 in list0:
        file_path = '../data/process01_' + l1
        print("开始合并文件:" + file_path)

        with open(file_path, encoding='utf-8') as file:
            for line in file:
                writer.writelines(line)
