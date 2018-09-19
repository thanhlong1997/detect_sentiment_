from pyvi import ViTokenizer
import numpy as np
import re
from operator import itemgetter
import data_utils as data_utils
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
def remove_stop_word(list):
    df = data_utils.read_excel('stopword.xlsx', sheetname='Sheet1', encoding="UTF-8")
    list2 = []
    for item in df.index:
        list2.append(df['stop'][item])
    for item in list:
        if item in list2:list.remove(item)
    # print(list2)

    return list    
