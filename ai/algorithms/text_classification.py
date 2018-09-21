import copy
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from operator import itemgetter
import heapq
import numpy as np
from ai.ai_utils.text_utils import get_normal_word, remove_stop_word

num_features_tag={'positive':150, 'negative':100,'neutral':120}
data_path= "D:\\ITSOL\\detect_sentiment_\\storage\\data\\data.xlsx"

def train_vectorizer(vocabulary, corpus):
    cou_vec = CountVectorizer(encoding='utf-8', tokenizer=get_normal_word, vocabulary=vocabulary)
    cou_vec.fit(corpus)
    return cou_vec

def train_MLP_title_prediction(tag, vectorizer, correct_titles, other_titles, MLP_structure=[64, 16], activation="tanh"):
    texts = copy.deepcopy(correct_titles)
    texts.append(tag)
    all_texts = copy.deepcopy(other_titles)
    all_texts.extend([text for text in texts if text not in all_texts])
    current_feature_gen = vectorizer
    current_clf = MLPClassifier(activation=activation, alpha=1e-4, batch_size="auto", beta_1=0.9,
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
    print(correct_labels)
    current_clf.fit(correct_vectors, correct_labels)
    current_model = {"feature_gen": current_feature_gen, "clf": current_clf}
    return current_model

def select_feature(trainning_set, tag):
    vocabulary = {}
    for sentence in trainning_set[tag]:
        sentence_tokenized = remove_stop_word(get_normal_word(sentence))
        for term in sentence_tokenized:
            if term not in vocabulary:
               vocabulary[term] = 1
            else:
               vocabulary[term] += 1
    features = [(key, value) for (key, value) in heapq.nlargest(num_features_tag[tag], vocabulary.items(), key=itemgetter(1))]
    return features

def build_vocab(list1,list2):
    list=list1
    for item in list2 :
        if list.count(item)<=0:
           list.append(item)
    return list

def build_vocabulary(training_data):
    vocab=[]
    for tag in training_data:
        print(len(training_data[tag]))
        for item in select_feature(training_data,tag):
            if vocab.count(item[0])<=0:vocab.append(item[0])
    corpus=[]
    df = pd.read_excel(data_path, sheetname='Sheet1', encoding="UTF-8")
    for item in df.index:
        corpus.append(df['content'][item])
    print("len vocab : ",len(vocab))
    return vocab,corpus    
