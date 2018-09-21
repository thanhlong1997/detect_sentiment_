from pyvi import ViTokenizer
import numpy as np
import re
from operator import itemgetter
from ai.ai_utils.data_utils import read_excel
pt= re.compile(r"_")
def segmentation(text):
    return ViTokenizer.tokenize(text)

def split_words(text):
    text = segmentation(text)
    try:
        return [x.strip("0123456789%@$.,=+-!;/()*\"&^:#|\n\t\'").lower() for x in text.split()]
    except TypeError:
        return []

def get_words_feature(text):
    split_word = split_words(text)
    return [word for word in split_word if word.encode('utf-8')]

def get_normal_word(text):
    get_words_featur=get_words_feature(text)
    list=[]
        # print(get_words_feature)
    for item in get_words_featur:
        sentence = re.sub(pt, " ", item)
        list.append(sentence)
    return list

def read_stopword(file):
    df = read_excel(file)
    list2 = []
    for item in df.index:
        list2.append(df['stop'][item])
    return list2

stopwords= read_stopword("D:\\ITSOL\\detect_sentiment_\\storage\\data\stopword.xlsx")

def remove_stop_word(list):
    for item in list:
        if item in stopwords:
            list.remove(item)
    return list    
