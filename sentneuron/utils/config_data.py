import os
import sys
import random

def load_pieces(datapath):
    pieces = []

    p_ix = 0
    for file in os.listdir(datapath):
        textpath = os.path.join(datapath, file)

        # Check if it is not a directory and if it has either .midi or .mid extentions
        if os.path.isfile(textpath) and (textpath[-4:] == ".txt"):
            filename = os.path.basename(textpath)
            pieces.append([])

            fp = open(textpath, "r")
            text = fp.read()

            # Each line represents one version of this piece
            for line in text.split("\n"):
                if len(line) > 0:
                    pieces[p_ix].append(line)

            p_ix += 1

    return pieces

def generate_shards(pieces, train_percent = 0.9):
    random.Random(42).shuffle(pieces)

    train_size = int(train_percent * len(pieces)) + 1

    train, test = [], []
    for i in range(train_size):
        random.Random(42).shuffle(pieces[i])
        for version in pieces[i]:
            train.append(version)
        print("train piece", i)

    for j in range(i, len(pieces)):
        random.Random(42).shuffle(pieces[j])
        for version in pieces[j]:
            test.append(version)

        print("test piece", j)

    return train, test

pieces_path = sys.argv[1]
pieces = load_pieces(pieces_path)
train,test = generate_shards(pieces)

# for p in pieces:
#     for version in p:
#         print("----------------------------")
#         print("----------------------------")
