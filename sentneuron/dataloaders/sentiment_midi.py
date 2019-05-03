import csv
import html
import math
import random
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

class SentimentMidi:
    def __init__(self, data_path, x_col_name, y_col_name, id_col_name, k=10, pad=False, balance=False):
        self.data_path = data_path

        self.data = self.load(data_path, x_col_name, y_col_name, id_col_name, pad)
        if balance:
            self.data = self.balance_dataset();

        xs = np.array([dp[0] for dp in self.data])
        ys = np.array([dp[1] for dp in self.data])

        posLabels = len(np.where(ys == 1.)[0])
        print("Total positive examples", posLabels)

        negLabels = len(np.where(ys == 0.)[0])
        print("Total negative examples", negLabels)

        self.split = StratifiedKFold(k, True, 42).split(xs, ys)

    def load(self, filepath, x_col_name, y_col_name, id_col_name, pad=False):
        csv_file = open(filepath, "r")
        data = csv.DictReader(csv_file)

        sentiment_data = []

        for row in data:
            x = row[x_col_name]
            if pad:
                max_len = self.find_longest_sequence_len(filepath, x_col_name)
                x = self.pad_sequence(x, max_len)

            y = int(float(row[y_col_name]))
            if y > 0:
                y = 1
            else:
                y = 0

            sentiment_data.append((x,y))

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

        for i in fold:
            # for sentence in self.data[i]:
                # print(sentence)
            piece, label = self.data[i]
            xs.append(piece)
            ys.append(label)

        return xs, ys

    def balance_dataset(self):
        xs = np.array([dp[0] for dp in self.data])
        ys = np.array([dp[1] for dp in self.data])

        # Separate majority and minority classes
        posLabels = np.where(ys == 1.)[0]
        negLabels = np.where(ys == 0.)[0]

        # Downsample majority class
        posLabels_downsampled = resample(xs[posLabels], replace=False, n_samples=len(negLabels), random_state=42)

        # Combine minority class with downsampled majority class
        balanced_data = []
        for i in range(len(posLabels_downsampled)):
            balanced_data.append((posLabels_downsampled[i], 1))

        for i in range(len(xs[negLabels])):
            balanced_data.append((xs[negLabels][i], 0))

        print(len(balanced_data))
        return balanced_data
