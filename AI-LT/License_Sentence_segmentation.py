#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2020/10/21 17:00
# software: PyCharm
from nltk import sent_tokenize
import os
import re
stop_phrases = ("The Initial Developer Grant", "Definitions","All rights reserved", "Copyright", "Fair Use Rights", "Preamble", "Grant of rights", "Section", "Version", "Grant of License", "License Grant", "Restrictions", "Additional", "Contributor", "Grant of Copyright License")

def get_text(file):
    with open (file,'r',encoding='unicode_escape') as f:
        text=""
        for line in f:
            text += line
    return text

def write_term():
    Folder_Path = u"..\\data\\test_license_sentences_file"  # 要拼接的文件夹及其完整路径
    license_list = u"..\\licensetestfile"
    file_list = os.listdir(license_list)

    for name in file_list:
        lf = os.path.join(license_list, name)
        corpus = get_text(lf)
        sentences = sent_tokenize(corpus)
        num=0
        filename, extension = os.path.splitext(name)
        for i in range(0, len(sentences)):
            l = len(str(sentences[i]))
            if l >= 30:
                if not sentences[i].startswith(stop_phrases, re.I):
                    num+=1
                    df = os.path.join(Folder_Path, filename+"_"+str(num)+extension)
                    with open(df, 'w', encoding='UTF-8') as f:
                        f.write(str(sentences[i]))

if __name__=='__main__':
    write_term()