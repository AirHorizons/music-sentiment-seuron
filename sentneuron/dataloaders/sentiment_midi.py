import csv
import html
import math
import random
import numpy as np

from sklearn.model_selection import KFold
from sklearn.utils import resample

class SentimentMidi:
    def __init__(self, data_path, x_col_name, y_col_name, id_col_name, pad=False, balance=False, separate_pieces=False, k=10):
        self.data_path = data_path

        self.data = self.load(data_path, x_col_name, y_col_name, id_col_name, pad)
        if balance:
            self.data = self.balance_dataset()

        ys = np.array([dp[2] for dp in self.data])
        posLabels = len(np.where(ys == 1.)[0])
        negLabels = len(np.where(ys == 0.)[0])

        print("Total positive examples", posLabels)
        print("Total negative examples", negLabels)

        self.separate_pieces = separate_pieces
        if self.separate_pieces:
            self.data = self.separate_ids()

        self.split = KFold(k, True, 42).split(self.data)

    def load(self, filepath, x_col_name, y_col_name, id_col_name, pad=False):
        csv_file = open(filepath, "r")
        data = csv.DictReader(csv_file)

        sentiment_data = []

        for row in data:
            id = row[id_col_name]

            x = row[x_col_name]
            if pad:
                max_len = self.find_longest_sequence_len(filepath, x_col_name)
                x = self.pad_sequence(x, max_len)

            y = int(float(row[y_col_name]))
            if y > 0:
                y = 1
            else:
                y = 0

            sentiment_data.append((id, x, y))

        csv_file.close()
        return sentiment_data

    def pad_sequence(self, sequence, max_len, pad_char='.'):
        padded_text = sequence.split(" ")
        padded_text += ['.'] * (max_len - len(padded_text))
        return " ".join(padded_text)

    def find_longest_sequence_len(self, filepath, x_col_name):
        csv_file = open(filepath, "r")

        max_len = 0
        data = csv.DictReader(csv_file)
        for row in data:
            x = row[x_col_name]
            if len(x.split(" ")) > max_len:
                max_len = len(x.split(" "))

        csv_file.close()
        return max_len

    def unpack_fold(self, fold):
        xs = []
        ys = []

        if self.separate_pieces:
            for i in fold:
                for sentence in self.data[i]:
                    id, x, y = sentence
                    xs.append(x)
                    ys.append(y)
        else:
            for i in fold:
                # for sentence in self.data[i]:
                    # print(sentence)
                id, x, y = self.data[i]
                xs.append(x)
                ys.append(y)

        return xs, ys

    def separate_ids(self):
        sentences_per_id = {}
        for piece in self.data:
            id, x, y = piece
            if id not in sentences_per_id:
                sentences_per_id[id] = []
            sentences_per_id[id].append((id, x, y))

        sentiment_data = []
        for p in sentences_per_id:
            sentiment_data.append(sentences_per_id[p])

        return sentiment_data

    def balance_dataset(self):
        ids = np.array([dp[0] for dp in self.data])
        xs  = np.array([dp[1] for dp in self.data])
        ys  = np.array([dp[2] for dp in self.data])

        # Separate majority and minority classes
        posIxs = np.where(ys == 1.)[0]
        negIxs = np.where(ys == 0.)[0]

        minClassIxs = negIxs
        maxClassIxs = posIxs

        if len(negIxs) > len(posIxs):
            minClassIxs = posIxs
            maxClassIxs = negIxs

        maxClassData = [(ids[i], xs[i], ys[i]) for i in maxClassIxs]
        minClassData = [(ids[i], xs[i], ys[i]) for i in minClassIxs]

        # Downsample majority class
        maxClassData_downsampled = resample(maxClassData, replace=False, n_samples=len(minClassData), random_state=42)

        # Combine minority class with downsampled majority class
        return maxClassData_downsampled + minClassData
