#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2020/10/30 17:29
# software: PyCharm

import gensim
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
# stop_text = open('stop_list.txt', 'r')
# stop_word = []
# for line in stop_text:
#     stop_word.append(line.strip())
TaggededDocument = gensim.models.doc2vec.TaggedDocument

file = "..\\data\\corpus_file\\corpus_result2.txt"

def get_corpus():

    with open(file, 'r',encoding="UTF-8") as doc:
        #         words_box = []
        # for line in doc:
        #     words_box.append(line.split())
        docs = doc.readlines()
    train_docs = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs

def train(x_train, size=100, epoch_num=70):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
    model_dm.save('model_doc2vec')
    return model_dm



if __name__ == '__main__':
    x_train = get_corpus()
    model_dm = train(x_train)
