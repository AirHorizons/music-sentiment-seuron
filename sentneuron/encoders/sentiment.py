import csv
import html

class SentimentData:
    def __init__(self, dir_path, x_col_name, y_col_name):
        dir_name = dir_path.split("/")[-1]
        self.test = self.load(dir_path + "/" + dir_name + "_test.csv", x_col_name, y_col_name)
        self.train = self.load(dir_path + "/" + dir_name + "_train.csv", x_col_name, y_col_name)
        self.validation = self.load(dir_path + "/" + dir_name + "_validation.csv", x_col_name, y_col_name)

    def load(self, filepath, x_col_name, y_col_name):
        csv_file = open(filepath, "r")
        data = csv.DictReader(csv_file)

        X = []
        Y = []
        for row in data:
            X.append(self.preprocess(row[x_col_name]))
            Y.append(row[y_col_name])

        return X, Y

    def preprocess(self, text, front_pad='\n ', end_pad=' '):
        text = html.unescape(text)
        text = text.replace('\n', ' ').strip()
        text = front_pad+text+end_pad
        # text = text.encode()
        return text
