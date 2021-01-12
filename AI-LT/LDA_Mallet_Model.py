#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2020/10/30 17:29
# software: PyCharm
import gensim
from gensim.corpora import BleiCorpus
from gensim import corpora
from pprint import pprint
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
import os

# Plotting tools
import matplotlib.pyplot as plt

class Corpus(object):
    def __init__(self, cursor, reviews_dictionary, corpus_path):
        self.cursor = cursor
        self.reviews_dictionary = reviews_dictionary
        self.corpus_path = corpus_path

    def __iter__(self):
      #  self.cursor.rewind()
        for review in self.cursor:
            yield self.reviews_dictionary.doc2bow(review)

    def serialize(self):
        BleiCorpus.serialize(self.corpus_path, self, id2word=self.reviews_dictionary)

        return self

class Dictionary(object):
    def __init__(self, cursor, dictionary_path):
        self.cursor = cursor
        self.dictionary_path = dictionary_path

    def build(self):
        # self.cursor.rewind()
        dictionary = corpora.Dictionary(self.cursor)
        # dictionary.filter_extremes(keep_n=1000000)
        dictionary.compactify()
        corpora.Dictionary.save(dictionary, self.dictionary_path)

        return dictionary


class Train:
    def __init__(self):
        pass

    @staticmethod
    def run(lda_model_path, corpus_path, num_topics, id2word):
        os.environ['MALLET_HOME'] = 'E:\\PythonProgram\\opensource\\mallet-2.0.8'
        mallet_path = 'E:\\PythonProgram\\opensource\\mallet-2.0.8\\bin\\mallet'  # update this path
        corpus = corpora.BleiCorpus(corpus_path)
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        # Show Topics
        pprint(ldamallet.show_topics(formatted=False))
        ldamallet.save(lda_model_path)

        return ldamallet

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    os.environ['MALLET_HOME'] = 'E:\\PythonProgram\\opensource\\mallet-2.0.8'
    mallet_path = 'E:\\PythonProgram\\opensource\\mallet-2.0.8\\bin\\mallet'  # update this path
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        ldamallet_model_path = "..\\data\\test_file\\ldamallet1\\lda_model_topics_"+ str(num_topics) + ".lda"
        model.save(ldamallet_model_path)

    return model_list, coherence_values

def write_to_file(words, file):
    with open(file, 'a+', encoding='utf-8')as f:
        f.write(str(words) + '\n')

def main():
    corpus_path = "..\\data\\test_file\\ldamallet1\\corpus.lda-c"
    dictionary_path = "..\\data\\test_file\\ldamallet1\\dictionary.dict"

    file = "..\\data\\corpus_file\\corpus_result2.txt"
    coherence_values_file = "..\\data\\test_file\\ldamallet1\\coherence_values_file.txt"
    with open(file,'r', encoding='utf-8') as f:
        words_box = []
        for line in f:
            words_box.append(line.split())

    dictionary = Dictionary(words_box, dictionary_path).build()
    # corpus = [dictionary.doc2bow(text) for text in words_box]
    Corpus(words_box, dictionary, corpus_path).serialize()
    corpus = corpora.BleiCorpus(corpus_path)

    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=words_box,start=500, limit=800, step=10)
    # Print the coherence scores
    # Show graph
    limit = 800
    start = 500
    step = 10
    x = range(start, limit, step)

    for m, cv in zip(x, coherence_values):
        s = "Num Topics ="+str(m)+" has Coherence Value of"+ str(round(cv, 8))
        write_to_file(s, coherence_values_file)

    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

if __name__ == '__main__':
    print("counting...")
    main()
    print("finish...")






