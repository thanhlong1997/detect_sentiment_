import copy
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from ai.ai_utils import text_utils
def select_feature(trainning_set, tag):
    vocabulary = {}
    for sentence in trainning_set[tag]:
        sentence_tokenized = text_utils.remove_stop_word(text_ultis.get_normal_word(sentence))
        for term in sentence_tokenized:
            if term not in vocabulary:
               vocabulary[term] = 1
            else:
               vocabulary[term] += 1
    number=0
    if tag=='positive':number=50
    if tag=='negative':number=30
    if tag=='neutral':number=20
    features = [(key, value) for (key, value) in heapq.nlargest(number, vocabulary.items(), key=itemgetter(1))]
    return features
def train_vectorizer(vocabulary, corpus):
    cou_vec = CountVectorizer(encoding='utf-8', tokenizer=get_normal_word, vocabulary=vocabulary)
    cou_vec.fit(corpus)
    return cou_vec    
def build_vocab(list1,list2):
        # vocab=[]
    list=list1
    for item in list2 :
        if list.count(item)<=0:
           list.append(item)
        # vocab+=list2
    return list
def build_vocabulary(training_data):
        # data = self.load_excel_training_data("data.xlsx", "Sheet1")
    vocab=[]
    for tag in training_data:
        print(len(training_data[tag]))
        for item in self.select_feature(training_data,tag):
            if vocab.count(item[0])<=0:vocab.append(item[0])
    corpus=[]
    df = pd.read_excel('data.xlsx', sheetname='Sheet1', encoding="UTF-8")
    for item in df.index:
        corpus.append(df['content'][item])
    print("len vocab : ",len(vocab))
    return vocab,corpus    
