import heapq
import re
from pyvi import ViTokenizer
import numpy as np
from operator import itemgetter
pt= re.compile(r"_")

# def preprocess(text):
#     sentence= re.sub(pt," ", text)
#     return sentence
# print(preprocess("Việt_nam"))
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
print(get_normal_word("Cộng hòa xã hội chủ nghĩa việt nam"))
def remove_stop_word(list):
    f = open("D:\\studying\\20172\\ML\\Data\\stop.txt", 'r', encoding="UTF-8")
    str1=str(f.read()).splitlines()
    for item in list:
        if item in str1: list.remove(item)
    str2=[u'năm',u'công_ty',u'đồng','việt_nam',u'và',u'là','usd',u'trong',u'số',u'các',u'đó',u'trong',u'vang',u'tỷ']
    for item in list:
        if item in str2:list.remove(item)
    # list.remove('đồng')
    df = pd.read_excel('stopword.xlsx', sheetname='Sheet1', encoding="UTF-8")
    list2 = []
    for item in df.index:
        list2.append(df['stop'][item])
    for item in list:
        if item in list2:list.remove(item)
    # print(list2)

    return list

def select_feature(list_sentences):
    vocabulary={}

    for sentence in list_sentences:
        sentence_tokenized=remove_stop_word(get_words_feature(sentence))
        for term in sentence_tokenized:
            if term not in vocabulary:
                vocabulary[term]=1
            else:
                vocabulary[term]+=1
    features= [(key, value) for (key, value) in heapq.nlargest(30,vocabulary.items(),key=itemgetter(1))]
    return features

# def content_keyword_analysis(titles):
#     vectorizer= CountVectorizer(ngram_range=(1,2), max_features=1000, stop_words=stopword, binary=True)
#     vector= vectorizer.fit_transform(titles).toarray()
#     total= np.sum(vector, axis= 0)
#     term_frequence= dict(zip(list(vectorizer.vocabulary_.keys()), [total[i] for i in  vectorizer.vocabulary_.values()]))
#     topitems = heapq.nlargest(1000, term_frequence.items(), key=itemgetter(1))
#     return topitems

import pandas as pd
import xlrd
# df = pd.read_excel('data.xlsx', sheetname='Sheet1',encoding="UTF-8")
# list=[]
# for item in df.index:
#     if int(df['label'][item])==1:
#         list.append(df['content'][item])
# # print(heapq.nlargest(1000,select_feature(list)['value']))
# # print([[word.value]for word in select_feature(list)])
# vocal=[]
# for item in select_feature(list):
#     vocal.append(get_normal_word(item[0])[0])
# print(vocal)
# # if u'và' in select_feature(list): print('yes')
# print(list)
# from sklearn.feature_extraction.text import CountVectorizer
# cou_vec=CountVectorizer(encoding='utf-8',tokenizer=get_normal_word,vocabulary=vocal)
# cou_vec.fit(list)
# output= cou_vec.transform(["Công ty A đạt tăng trưởng ở mức cao"])
# print(output)
# print(get_normal_word("Công ty A đạt tăng trưởng ở mức cao"))
import copy
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
tags={1:'positive', 0:'neutral', -1:'negative'}
from sklearn.model_selection import train_test_split
class MLP(object):
    def train_MLP_title_prediction(self,tag, vectorizer, correct_titles, other_titles, MLP_structure=[64, 16]):
        texts = copy.deepcopy(correct_titles)
        texts.append(tag)
        all_texts = copy.deepcopy(other_titles)
        all_texts.extend([text for text in texts if text not in all_texts])
        current_feature_gen = vectorizer
        current_clf = MLPClassifier(activation="tanh", alpha=1e-4, batch_size="auto", beta_1=0.9,
                                    beta_2=0.999, early_stopping=False, epsilon=1e-8,
                                    hidden_layer_sizes=(MLP_structure[0], MLP_structure[1],), learning_rate="constant",
                                    learning_rate_init=0.01, max_iter=500, momentum=0.8,
                                    nesterovs_momentum=True, power_t=0.5, random_state=21,
                                    shuffle=True, solver='adam', tol=1e-4, validation_fraction=0.1,
                                    verbose=False, warm_start=False)
    # current_clf= SVC(probability=True)
        correct_vectors = current_feature_gen.transform(texts).toarray().tolist()
        correct_vectors = [vector for vector in correct_vectors if np.any(vector)]
        correct_labels = [tag] * len(correct_vectors)
        all_vectors = current_feature_gen.transform(all_texts).toarray().tolist()
        other_vectors = [vec for vec in all_vectors if vec not in correct_vectors]
        zeros=[[0]*len(other_vectors[0])]*4
        other_vectors.extend(zeros)
        other_labels = ['other'] * len(other_vectors)
        correct_vectors.extend(other_vectors)
        correct_labels.extend(other_labels)
        # print(correct_labels)
        # print(len(correct_labels))
        # print(len(correct_vectors))
        # print("accuracy ",len(correct_labels))
        count=0
        for item in correct_labels:
            if item==tag:count+=1
        # print(count/len(correct_labels))
        # X_train,X_test,Y_train,Y_test=train_test_split( correct_vectors, correct_labels, test_size=0.33, random_state=42)
        current_clf.fit(correct_vectors, correct_labels)
        # print(correct_vectors,"\n" ,correct_labels)
        # print(current_clf.score(X_test,X_test))
        current_model = {"feature_gen": current_feature_gen, "clf": current_clf}
        return current_model

    def train_vectorizer(self, vocabulary, corpus):
        cou_vec = CountVectorizer(encoding='utf-8', tokenizer=get_normal_word, vocabulary=vocabulary)
        cou_vec.fit(corpus)
        return cou_vec


    def load_excel_training_data(self,excel_file,excel_sheet):
        df = pd.read_excel(excel_file, sheetname=excel_sheet, encoding="UTF-8")
        training_data={}
        for index in tags:
            training_data[tags[index]]=[]

        for item in df.index:
            training_data[tags[df['label'][item]]].append(df['content'][item])
        return training_data

    # for tag in self.data:
    #     print(tag,"---",data[tag])
# print(1/0)
# print(load_excel_training_data("data.xlsx","Sheet1"))
    def build_vocab(self,list1,list2):
        # vocab=[]
        for item in list2 :
            if list1.count(item)<=0:
                list1.append(item)
        # vocab+=list2
        return list1
    def build_vocabulary(self,training_data):
        # data = self.load_excel_training_data("data.xlsx", "Sheet1")
        vocab=[]
        for tag in tags:
            for item in training_data[tags[tag]]:
                vocab=self.build_vocab(vocab,get_normal_word(item))
        corpus=[]
        df = pd.read_excel('data.xlsx', sheetname='Sheet1', encoding="UTF-8")
        # for tag in tags:
        # for index in tags:
        #     corpus+=training_data[tags[index]]
        for item in df.index:
            corpus.append(df['content'][item])
        return vocab,corpus
    def train_clf(self, excel_file, excel_sheet="Sheet1"):
        self.tag_model = {}
        training_data = self.load_excel_training_data(excel_file, excel_sheet)
        testing_data={}
        # print(training_data)
        for tag in training_data:
            training_data[tag]=np.asarray(training_data[tag])
            train_numbers = np.random.choice(training_data[tag].shape[0], round(training_data[tag].shape[0] * 0.80), replace=False)
            test_numbers = np.array(list(set(range(training_data[tag].shape[0])) - set(train_numbers)))
            testing_data[tag]=list(training_data[tag][test_numbers])
            training_data[tag]=list(training_data[tag][train_numbers])
            print(len(training_data[tag]),len(testing_data[tag]))


        vocabulary, corpus = self.build_vocabulary(training_data)
        self.vectorizer = self.train_vectorizer(vocabulary, corpus)
        for tag in training_data:
            print("train content model for tag: ", tag)
            tag_excluded_data_dict = copy.deepcopy(training_data)
            texts = copy.deepcopy(training_data[tag])
            print(len(texts))
            del tag_excluded_data_dict[tag]
            other_texts = []
            for other_tag in tag_excluded_data_dict:
                other_texts.extend(tag_excluded_data_dict[other_tag])
            print(len(other_texts))
            curr_clf = self.train_MLP_title_prediction(tag, self.vectorizer, texts, other_texts, MLP_structure=[4, 256])
            # print(curr_clf.predict(texts))
            self.tag_model[tag] = curr_clf
            print(tag)
            # print(self.tag_model[tag]['clf'])
        test_data=[]
        test_label=[]
        corect={}
        score={}
        label_predict={}
        for index in tags:
            label_predict[tags[index]]=[]
        for tag in testing_data:
            print(self.tag_model[tag]['clf'].predict(self.tag_model[tag]['feature_gen'].transform(['Công  ty phát triển 10%'])))
            # print(len(testing_data[tag]))
            test_data+=testing_data[tag]
            for item in testing_data[tag]:
                test_label.append(tag)
        print(len(test_data),len(test_label))
        print(test_label)
        for tag in testing_data:
            # print(tag)
            corect[tag]=self.tag_model[tag]['clf'].predict(self.tag_model[tag]['feature_gen'].transform(test_data))
            score[tag]=self.tag_model[tag]['clf'].predict_proba(self.tag_model[tag]['feature_gen'].transform(test_data))
            score[tag]=np.amax(score[tag],axis=1)
            print(score[tag].shape)
        print(corect)
        print(score)
        # print(corect['negative'][0])
        for i in range(68):
            tag1=''
            tag2=''
            tag3=''
            a=max(score['positive'][i],score['neutral'][i],score['negative'][i])
            if score['positive'][i]==a:tag1='positive'
            if score['negative'][i] == a: tag1 = 'negative'
            if score['neutral'][i] == a: tag1 = 'neutral'
            b=min(score['positive'][i],score['neutral'][i],score['negative'][i])
            if score['positive'][i]==b:tag3='positive'
            if score['negative'][i] == b: tag3 = 'negative'
            if score['neutral'][i] == b: tag3 = 'neutral'
            if (tag1=='positive')&(tag3=='negative'):tag2='neutral'
            if (tag1 == 'positive') & (tag3 == 'neutral'): tag2 = 'negative'
            if (tag1 == 'negative') & (tag3 == 'neutral'): tag2 = 'positive'
            if (tag1 == 'negative') & (tag3 == 'positive'): tag2 = 'neutral'
            if (tag1 == 'neutral') & (tag3 == 'negative'): tag2 = 'positive'
            if (tag1 == 'neutral') & (tag3 == 'positive'): tag2 = 'negative'
            if corect[tag1][i]!='other':
                label_predict[test_label[i]].append(corect[tag1][i])
                continue
            if corect[tag2][i]!='other':
                label_predict[test_label[i]].append(corect[tag2][i])
                continue
            if corect[tag3][i]!='other':
                label_predict[test_label[i]].append(corect[tag3][i])
                continue
            label_predict['neutral'].append('neutral')

        print(label_predict)
        for tag in label_predict:
            sum = 0
            for item in label_predict[tag]:
                if item==tag:sum+=1
            print("acc of ", tag ,": ", sum/len(label_predict[tag]))
        # print(label_predict)


mlp= MLP()
train=mlp.train_clf("data.xlsx","Sheet1")
print
# train.predict()
# for index in tags:
#     print((index))
# train=MLP().load_excel_training_data("data.xlsx",'Sheet1')
# for item in train:
#     print(type(item),item)
#     print(train[item])
