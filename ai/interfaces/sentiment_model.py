import copy
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
tags={1:'positive', 0:'neutral', -1:'negative'}
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


    def train_clf(self, excel_file, excel_sheet="Sheet1"):
        self.tag_model = {}
        training_data = load_excel_training_data(excel_file, excel_sheet)
        testing_data={}
        # print(training_data)
        for tag in training_data:
            training_data[tag]=np.asarray(training_data[tag])
            train_numbers = np.random.choice(training_data[tag].shape[0], round(training_data[tag].shape[0] * 0.80), replace=False)
            test_numbers = np.array(list(set(range(training_data[tag].shape[0])) - set(train_numbers)))
            testing_data[tag]=list(training_data[tag][test_numbers])
            training_data[tag]=list(training_data[tag][train_numbers])
            print(len(training_data[tag]),len(testing_data[tag]))
        training_data['negative'].append('Trong năm 2017 công ty tăng trưởng chậm so với cùng kỳ năm trước')
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
            if score['negative'][i]==b:tag3='negative'
            if score['neutral'][i] == b: tag3 = 'neutral'
            if score['positive'][i] == b: tag3 = 'positive'
            if (tag1 == 'positive')& (tag3 == 'negative'):tag2 = 'neutral'
            if (tag1 == 'positive') & (tag3 == 'neutral'): tag2 = 'negative'
            if (tag1 == 'negative') & (tag3 == 'neutral'): tag2 = 'positive'
            if (tag1 == 'negative') & (tag3 == 'positive'): tag2 = 'neutral'
            if (tag1 == 'neutral') & (tag3 == 'negative'): tag2 = 'positive'
            if (tag1 == 'neutral') & (tag3 == 'positive'): tag2 = 'negative'
            print(tag1,tag2,tag3)
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
        fail=0
        for tag in label_predict:
            sum = 0
            for item in label_predict[tag]:
                if item==tag:sum+=1
                if item!=tag:fail+=1
            print("acc of ", tag ,": ", sum/len(label_predict[tag]))
        print("average : ", (68-fail)/68)
        print(68-fail)
        # print(label_predict)
        for tag in label_predict:
            print(self.tag_model[tag]['clf'].predict(self.tag_model[tag]['feature_gen'].transform(['công ty phát triển chậm'])))
