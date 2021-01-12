#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2020/10/15 16:08
# software: PyCharm
import sys,re,collections,nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import gensim
import os

# 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
pat_letter = re.compile(r'[^a-zA-Z \']+')
# 还原常见缩写单词
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
pat_s = re.compile("(?<=[a-zA-Z])\'s") # 找出字母后面的字母
pat_s2 = re.compile("(?<=s)\'s?")
pat_not = re.compile("(?<=[a-zA-Z])n\'t") # not的缩写
pat_would = re.compile("(?<=[a-zA-Z])\'d") # would的缩写
pat_will = re.compile("(?<=[a-zA-Z])\'ll") # will的缩写
pat_am = re.compile("(?<=[I|i])\'m") # am的缩写
pat_are = re.compile("(?<=[a-zA-Z])\'re") # are的缩写
pat_ve = re.compile("(?<=[a-zA-Z])\'ve") # have的缩写
pat_url = re.compile(r'^((https|http|ftp|rtsp|mms)?:\/\/)[^\s]+')

lmtzr = WordNetLemmatizer()

def replace_abbreviations(text):
    new_text = pat_letter.sub(' ', text).strip().lower()
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_url.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    new_text = re.sub(r"[\w]+@[\.\w]+", "", new_text)  # 邮件地址，没意义
    new_text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", new_text)  # 网址，没意义

    return new_text

# 词性还原之词性选择：pos和tag有相似的地方，通过tag获得pos
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

# 停顿词
def load_stopwords():
    stopwords = {}
    with open('..\\data\\corpus_file\\stopwords.txt', 'rU') as f:
        for line in f:
            stopwords[line.strip()] = 1
    return stopwords

# Build the bigram and trigram models
def Build_bigram_trigram(words_box):
    bigram = gensim.models.Phrases(words_box, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[words_box], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return  trigram_mod[bigram_mod[words_box]]


# 词形还原
def merge(words):
    nouns =[]
    stopwords = load_stopwords()
    for word in words:
        if word:
            tokens = word_tokenize(word)

            text = [wd for wd in tokens if wd not in stopwords] # 去除停顿词
            text = Build_bigram_trigram(text)  # 建立双单词和三单词短语模型
            tagged_text = nltk.pos_tag(text)  # 获得每一个单词的词性

            for wd, tag in tagged_text:
                pos = get_wordnet_pos(tag)
                if pos:
                    nouns.append(lmtzr.lemmatize(wd,pos)) #还原单词
                else:
                    nouns.append(wd)
    return nouns

# 将统计结果写入文件
def write_to_file(words, file):
    dir_way = "..\\data\\sentence_data_file\\"+ file
    f = open(dir_way, 'a+',encoding='unicode_escape')
    for item in words:
        f.write(item + ' ')

def get_words(str,file):
    words_box = []
    words_box.extend(merge(replace_abbreviations(str).split()))
    write_to_file(words_box,file)

if __name__=='__main__':
    print ("counting...")
    sentence_file_name = "..\\data\\license_sentences_file1"  #许可证句子文件夹目录
    files = os.listdir(sentence_file_name)  # 得到文件夹下的所有文件名称

    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = open(sentence_file_name + "/" + file, encoding='unicode_escape')  # 打开文件
            iter_f = iter(f)  # 创建迭代器
            str = ""
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                str = str + line
            get_words(str,file)

    # get_words(corpus_file_name)
    print ("writing file...")


