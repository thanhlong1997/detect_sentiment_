import pickle
from ai.algorithms.text_classification import *
from ai.ai_utils.data_utils import *


class Classifier(object):
    def __init__(self):
        pass

    def train_clf(self, excel_file, excel_sheet="Sheet1"):
        self.tag_model = {}
        training_data = load_excel_training_data(excel_file, excel_sheet)
        training_data, testing_data= split_data(training_data, 0.8)
        vocabulary, corpus = build_vocabulary(training_data)
        self.vectorizer = train_vectorizer(vocabulary, corpus)
        for tag in training_data:
            print("train model for tag: ", tag)
            tag_excluded_data_dict = copy.deepcopy(training_data)
            texts = copy.deepcopy(training_data[tag])
            print(len(texts))
            del tag_excluded_data_dict[tag]
            other_texts = []
            for other_tag in tag_excluded_data_dict:
                other_texts.extend(tag_excluded_data_dict[other_tag])
            print(len(other_texts))
            curr_clf = train_MLP_title_prediction(tag, self.vectorizer, texts, other_texts, MLP_structure=[4, 256])
            # print(curr_clf.predict(texts))
            self.tag_model[tag] = curr_clf
        print("training completed. testing . . . . . .")
        self.predict_test_data(testing_data)
        pass

    def dump(self, storage_path):
        with open(storage_path, "wb") as f:
            pickle.dump(self.tag_model, f)
            f.close()

    def load(self, storage_path):
        with open(storage_path, "rb") as f:
            self.tag_model= pickle.load(f)
            f.close()

    def predict_tag(self, tag, text):
        vector= self.tag_model[tag]['feature_gen'].transform([text])
        pred = self.tag_model[tag]['first_level']['clf'].predict(vector)[0]
        prob = max(self.tag_model[tag]['first_level']['clf'].predict_proba(vector)[0])
        if pred == 'other':
            return 0
        return prob

    def predict(self, text):
        max_prob=0
        pred="positive"
        for tag in tags:
            prob= self.predict_tag(tag, text)
            if prob > max_prob:
                max_prob= prob
                pred= tag
        return pred

    def predict_test_data(self,testing_data):
        test_data = []
        test_label = []
        sum={'average':0}
        for tag in testing_data:
            test_data.extend( testing_data[tag])
            test_label.extend([tag]*len(testing_data[tag]))
            sum[tag]=0
        for i in range(len(test_data)):
            predicted_tag= self.predict(test_data[i])
            if predicted_tag == test_data[i]:
                sum['average']+=1
                sum[predicted_tag]+=1
        print("average acc: ", sum['average']/len(test_data))
        for tag in tags:
            print(tag," acc: ", sum[tag]/len(testing_data[tag]))