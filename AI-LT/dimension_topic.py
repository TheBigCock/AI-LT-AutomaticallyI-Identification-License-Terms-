#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2020/10/30 20:19
# software: PyCharm

import pandas as pd
from pprint import pprint
import gensim
from gensim.corpora import BleiCorpus
from gensim import corpora

import os
from gensim.models.wrappers import LdaMallet
import csv

def write_to_file(words, file = "..\\data\\test_file\\d_f.txt"):
    with open(file, 'a+', encoding='utf-8')as f:
        f.write(words + '\n')

def get_sentence_topic():

    # 展示模型结果
    dictionary_path = "..\\data\\test_file\\ldamallet1\\dictionary.dict"
    lda_model_path = "..\\data\\test_file\\ldamallet1\\lda_model_topics_790.lda"


    dictionary = corpora.Dictionary.load(dictionary_path)
    # corpus = corpora.BleiCorpus(corpus_path)
    # corpus = [dictionary.doc2bow(text) for text in s]

    model_lda = gensim.models.ldamodel.LdaModel.load(lda_model_path)
    ldamallet = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_lda)

    dimension_file_name = "..\\data\\test_file\\license_dimension_participle"  # 文件夹目录
    files = os.listdir(dimension_file_name)  # 得到文件夹下的所有文件名称

    dimension_topic = {}
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            # f = open(dimension_file_name + "/" + file, encoding='unicode_escape')  # 打开文件
            words_box = []
            with open(dimension_file_name + "/" + file, 'r') as d_file:
                for line in d_file:  # 遍历文件，一行行遍历，读取文本
                    words_box.append(line.split())

            corpus_twitter = [dictionary.doc2bow(text) for text in words_box]
            # topics_twitter = ldamallet.get_document_topics(corpus_twitter)
            # print("主题分布分别为：\n", topics_twitter[0], "\n", topics_twitter[1],"\n",topics_twitter[2])
            t = []
            for i, row in enumerate(ldamallet[corpus_twitter]):
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j==0:
                        t.append(topic_num)
            t = list(set(t))
            t.sort()
            (path, filename) = os.path.splitext(file)
            s = path +":"+str(t)
            # dimension_topic[path] = t
            write_to_file(s)
    return dimension_topic
    # print(dimension_topic)

if __name__ == "__main__":
    get_sentence_topic()

