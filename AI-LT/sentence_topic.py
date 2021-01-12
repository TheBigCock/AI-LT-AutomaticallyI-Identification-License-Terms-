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

path = "..\\data\\license_sentences_file1" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
s = []
for file in files: #遍历文件夹
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
          f = open(path+"/"+file, encoding='unicode_escape') #打开文件
          iter_f = iter(f) #创建迭代器
          str = ""
          for line in iter_f: #遍历文件，一行行遍历，读取文本
              str = str + line
          s.append(str) #每个文件的文本存到list中


#展示模型结果
corpus_path = "..\\data\\test_file\\ldamallet1\\corpus.lda-c"
dictionary_path = "..\\data\\test_file\\ldamallet1\\dictionary.dict"
lda_num_topics = 790
lda_model_path = "..\\data\\test_file\\ldamallet1\\lda_model_topics_790.lda"
# file = "..\\data\\test_file\\corpusLicense.txt"

dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
# corpus = [dictionary.doc2bow(text) for text in s]

model_lda = gensim.models.ldamodel.LdaModel.load(lda_model_path)
ldamallet = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model_lda)
texts = s
# ldamallet = LdaMallet.load(lda_model_path)

def format_topics_sentences(ldamalletmodel=ldamallet, corpus=corpus,texts = texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamalletmodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamalletmodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamalletmodel=ldamallet, corpus=corpus)
# df_topic_sents_keywords.to_csv("sentence_topic_data.csv")

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_topic_sents_keywords.to_csv("sentence_topic_data.csv")

# Show
print(df_dominant_topic.head(10))



# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
sent_topics_sorteddf_mallet.to_csv("sent_topics_sorteddf_mallet.csv")
# Show
sent_topics_sorteddf_mallet.head()