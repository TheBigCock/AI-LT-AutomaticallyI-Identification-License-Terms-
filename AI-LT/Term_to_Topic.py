#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2020/11/8 10:19
# software: PyCharm


import os
from os import listdir

import gensim
import numpy as np
from gensim import corpora

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
            # s = path +":"+str(t)
            dimension_topic[path] = t
            # write_to_file(s)
    return dimension_topic

def train():
    """训练 Doc2Vec 模型
    """

    # 先把所有文档的路径存进一个 array中，docLabels：
    data_dir = "..\\data\\test_file\\topic_participle"
    docLabels = [f for f in listdir(data_dir) if f.endswith('.txt')]

    data = []
    for doc in docLabels:
        ws = open(data_dir + "/" + doc, 'r', encoding='UTF-8').read()
        data.append(ws)

    print(len(data))
    # 训练 Doc2Vec，并保存模型：
    sentences = LabeledLineSentence(data, docLabels)
    # an empty model
    model = gensim.models.Doc2Vec(vector_size=100, window=3, min_count=1,
                                  workers=4, alpha=0.025, min_alpha=0.025, epochs=80)
    model.build_vocab(sentences)
    print("开始训练...")
    model.train(sentences, total_examples=model.corpus_count, epochs=80)

    model.save("..\\data\\test_file\\model\\doc2vec.model")
    print("model saved")


def pridict_model():
    print("load model")
    model = gensim.models.Doc2Vec.load('..\\data\\test_file\\model\\doc2vec.model')

    topic_file_name = "..\\data\\test_file\\topic_participle"  # 主题文件夹目录
    dimension_file_name = "..\\data\\test_file\\license_dimension_participle"  #许可证维度文件夹目录
    topic_files = listdir(topic_file_name)  # 得到文件夹下的所有文件名称
    dimension_files = listdir(dimension_file_name)  # 得到文件夹下的所有文件名称

    dimension_topic = {}
    for topic_file in topic_files:  # 遍历文件夹
        if not os.path.isdir(topic_file):  # 判断是否是文件夹，不是文件夹才打开
            st1 = open(topic_file_name + "/" + topic_file, 'r', encoding='UTF-8').read()
            # 转成句子向量
            vect1 = sent2vec(model, st1)
            dimension_cos = {}
            for dimension_file in dimension_files:
                if not os.path.isdir(dimension_file):
                    st2 = open(dimension_file_name + "/" + dimension_file, 'r', encoding='UTF-8').read()
                    # 转成句子向量
                    vect2 = sent2vec(model, st2)

                    # 查看变量占用空间大小
                    # import sys
                    # print(sys.getsizeof(vect1))
                    # print(sys.getsizeof(vect2))

                    cos = similarity(vect1, vect2)
                    (path, filename) = os.path.splitext(dimension_file)
                    dimension_cos[path] = cos
            dimension_value = max(dimension_cos, key=dimension_cos.get)
            # max(dimension_cos.values())
            num = topic_file[6:-4]
            if dimension_value in dimension_topic:
                dimension_topic.setdefault(dimension_value,[]).append(num)
            else:
                dimension_topic[dimension_value] = [num]
            # print("{}".format(dimension_topic.items()))
    return dimension_topic

def similarity(a_vect, b_vect):
    """计算两个向量余弦值

    Arguments:
        a_vect {[type]} -- a 向量
        b_vect {[type]} -- b 向量

    Returns:
        [type] -- [description]
    """

    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a * b
        a_norm += a ** 2
        b_norm += b ** 2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm * b_norm) ** 0.5)

    return cos


def sent2vec(model, words):
    """文本转换成向量

    Arguments:
        model {[type]} -- Doc2Vec 模型
        words {[type]} -- 分词后的文本

    Returns:
        [type] -- 向量数组
    """

    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum())


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])


if __name__ == '__main__':

    # train()

    dimension_topic = pridict_model()
    # print(dimension_topic)
    dimension_topic_dlm = get_sentence_topic()
    # print(dimension_topic_dlm)
    with open("..\\data\\test_file\\dimension_to_topic2.txt",'w+',encoding='utf-8') as dt:
        for k,v in dimension_topic.items():
            v = list(set(v))
            v = list(map(int, v))
            v.sort()
            print(str(k) + ":" + str(v))
            print("合并后：\n")
            if k in dimension_topic_dlm:
                d_t = dimension_topic_dlm[k]+v
            v = list(set(d_t))
            v.sort()
            dt.write(str(k) + ":" + str(v) +'\n')
            print(str(k) + ":" + str(v))