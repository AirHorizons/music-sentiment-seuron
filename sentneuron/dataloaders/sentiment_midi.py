import csv
import html
import math
import random

from sklearn.model_selection import KFold

class SentimentMidi:
    def __init__(self, data_path, x_col_name, y_col_name, id_col_name, k=10, pad=False):
        self.data_path = data_path
        self.data = self.load(data_path, x_col_name, y_col_name, id_col_name, pad)
        self.split = KFold(k, True, 42).split(self.data)

    def load(self, filepath, x_col_name, y_col_name, id_col_name, pad=False):
        csv_file = open(filepath, "r")

        max_len = 0
        data = csv.DictReader(csv_file)
        for row in data:
            x = row[x_col_name]
            if len(x.split(" ")) > max_len:
                max_len = len(x.split(" "))

        sentiment_data = []
        for row in data:
            # If piece id is not in dictionary, add to it
            # id = row[id_col_name]
            # if id not in sentiment_data:
            #     sentiment_data[id] = []

            # Parse sentence x and label y

            x = row[x_col_name]
            if pad:
                s_text = x.split(" ")
                s_text += ['.'] * (max_len - len(s_text))
                x = " ".join(s_text)

            y = int(float(row[y_col_name]))
            if y > 0:
                y = 1
            else:
                y = 0

            sentiment_data.append((x,y))

        csv_file.close()
        return sentiment_data



        #
        #     sentiment_data[id].append((x, y))
        #
        # # Map dictionary to list of padded (same width) sentences
        # sentences_per_piece = []
        # for p in sentiment_data:
        #     sentences = []
        #     for s in sentiment_data[p]:
        #         text, label = s
        #         if pad:
        #             s_text = text.split(" ")
        #             s_text += ['.'] * (max_len - len(s_text))
        #
        #             sentences.append((" ".join(s_text), label))
        #         else:
        #             sentences.append((text, label))
        #
        #     sentences_per_piece.append(sentences)

        # return sentences_per_piece

    def unpack_fold(self, fold):
        xs = []
        ys = []

        for i in fold:
            # for sentence in self.data[i]:
                # print(sentence)
                piece, label = self.data[i]
                xs.append(piece)
                ys.append(label)

        # unpacked_fold = list(zip(xs, ys))

        # random.seed(42)
        # random.shuffle(unpacked_fold)
        #
        # xs, ys = zip(*unpacked_fold)

        return xs, ys
