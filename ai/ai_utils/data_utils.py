import pandas as pd
import numpy as np
tags={1:'positive', 0:'neutral', -1:'negative'}

def load_excel_training_data(excel_file,excel_sheet):
    df = pd.read_excel(excel_file, sheetname=excel_sheet, encoding="UTF-8")
    training_data={}
    for index in tags:
        training_data[tags[index]]=[]
    for item in df.index:
        training_data[tags[df['label'][item]]].append(df['content'][item])
    return training_data

def read_excel(excel_file,excel_sheet= "Sheet1"):
    df=pd.read_excel(excel_file, sheetname=excel_sheet, encoding="UTF-8")
    return df

def split_data(training_data, ratio):
    testing_data = {}
    for tag in training_data:
        training_data[tag] = np.asarray(training_data[tag])
        train_numbers = np.random.choice(training_data[tag].shape[0], round(training_data[tag].shape[0] * ratio),
                                         replace=False)
        test_numbers = np.array(list(set(range(training_data[tag].shape[0])) - set(train_numbers)))
        testing_data[tag] = list(training_data[tag][test_numbers])
        training_data[tag] = list(training_data[tag][train_numbers])
    return training_data, testing_data