from ai.interfaces.sentiment_model import Classifier


data_path= "D:\\ITSOL\\detect_sentiment_\\storage\\data\\data.xlsx"
model_path= "D:\\ITSOL\\detect_sentiment_\\storage\\model\\sentiment_model.pkl"

classifier= Classifier()
classifier.train_clf(data_path)
classifier.dump(model_path)
classifier.load(model_path)
print(classifier.predict("Công ty vừa đạt lợi nhuận cao trong năm ngoái"))