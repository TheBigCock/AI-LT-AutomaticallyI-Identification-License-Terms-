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

def write_to_file(words, file = "..\\data\\test_file\\s_f.txt"):
    with open(file, 'a+', encoding='utf-8')as f:
        f.write(str(words) + '\n')

def get_sentence_topic():

    # 展示模型结果
    dictionary_path = "..\\data\\test_file\\ldamallet1\\dictionary.dict"
    lda_model_path = "..\\data\\test_file\\ldamallet1\\lda_model_topics_790.lda"
    file = "..\\data\\corpus_file\\corpus_result2.txt"

    dictionary = corpora.Dictionary.load(dictionary_path)
    # corpus = corpora.BleiCorpus(corpus_path)
    # corpus = [dictionary.doc2bow(text) for text in s]

    model_lda = gensim.models.ldamodel.LdaModel.load(lda_model_path)
    ldamallet = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_lda)

    with open(file, 'r', encoding='utf-8') as f:
        words_box = []
        for line in f:
            words_box.append(line.split())

    corpus_twitter = [dictionary.doc2bow(text) for text in words_box]
    # topics_twitter = ldamallet.get_document_topics(corpus_twitter)

    sentence_file_name = "..\\data\\license_sentences_file1"  # 文件夹目录
    files = os.listdir(sentence_file_name)  # 得到文件夹下的所有文件名称

    sentence_list = []
    for s_file in files:  # 遍历文件夹
        if not os.path.isdir(s_file):  # 判断是否是文件夹，不是文件夹才打开
            f = sentence_file_name + "\\" + s_file
            sentence_list.append(str(f))

    print(len(sentence_list))


    for i, row in enumerate(ldamallet[corpus_twitter]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                s = str(topic_num) +", " + str(prop_topic) +", "+ str(sentence_list[i])
                # wp = ldamallet.show_topic(topic_num)
                # topic_keywords = ", ".join([word for word, prop in wp])
                write_to_file(s)

def txt_to_csv():
    s_f = "..\\data\\test_file\\s_f.txt"
    with open('..\\data\\test_file\\s_f.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        # 读要转换的txt文件，文件每行各词间以,字符分隔
        with open(s_f, 'r') as filein:
            for line in filein:
                line_list = line.strip('\n').split(',')
                spamwriter.writerow(line_list)

if __name__ == "__main__":
    s_f = "..\\data\\test_file\\s_f.csv"
    result = {}
    header = ['topic', 'prec', 'dir']
    with open(s_f, 'r') as f:
        red = csv.DictReader(f,header)
        for d in red:
            result.setdefault(d['topic'], []).append(d['dir'])

    for i in sorted(result):
        some_topic_path = "..\\data\\test_file\\topic\\topic_" + str(i) + ".txt"
        with open(some_topic_path,"a+", encoding='utf-8') as s_t_f:
            for j in range(len(result[i])):
                with open(str(result[i][j]).lstrip(), 'r', encoding='unicode_escape') as rf:
                    str1 = ' '.join(str(line) for line in rf)
                print(str(j)+':'+result[i][j])
                s_t_f.write(str1+"\n")

