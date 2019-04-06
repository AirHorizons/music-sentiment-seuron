import csv
import html
import math
import random

from sklearn.model_selection import KFold

class SentimentMidi:
    def __init__(self, data_path, x_col_name, y_col_name, id_col_name, k=10):
        self.data_path = data_path
        self.data = self.load(data_path, x_col_name, y_col_name, id_col_name)
        self.split = KFold(k, True, 42).split(self.data)

    def load(self, filepath, x_col_name, y_col_name, id_col_name):
        csv_file = open(filepath, "r")
        data = csv.DictReader(csv_file)

        sentiment_data = {}

        max_len = 0

        for row in data:
            # If piece id is not in dictionary, add to it
            id = row[id_col_name]
            if id not in sentiment_data:
                sentiment_data[id] = []

            # Parse sentence x and label y
            x = row[x_col_name]
            y = int(float(row[y_col_name]))
            if y > 0:
                y = 1
            else:
                y = 0

            if len(x.split(" ")) > max_len:
                max_len = len(x.split(" "))

            sentiment_data[id].append((x, y))

        # Map dictionary to list of padded (same width) sentences
        sentences_per_piece = []
        for p in sentiment_data:
            sentences = []
            for s in sentiment_data[p]:
                text, label = s

                text = text.replace('\n', '')[:-1]
                s_text = text.split(" ")
                s_text += ['.'] * (max_len - len(s_text))

                sentences.append((" ".join(s_text), label))
            sentences_per_piece.append(sentences)

        return sentences_per_piece

    def unpack_fold(self, fold):
        xs = []
        ys = []

        for i in fold:
            for sentence in self.data[i]:
                piece, label = sentence
                xs.append(piece)
                ys.append(label)

        unpacked_fold = list(zip(xs, ys))

        random.seed(42)
        random.shuffle(unpacked_fold)

        xs, ys = zip(*unpacked_fold)

        return list(xs), list(ys)
