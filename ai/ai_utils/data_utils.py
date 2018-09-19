    def load_excel_training_data(self,excel_file,excel_sheet):
        df = pd.read_excel(excel_file, sheetname=excel_sheet, encoding="UTF-8")
        training_data={}
        for index in tags:
            training_data[tags[index]]=[]

        for item in df.index:
            training_data[tags[df['label'][item]]].append(df['content'][item])
        return training_data
