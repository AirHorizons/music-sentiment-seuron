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

def split_data(pieces, train_percent = 0.9):
    random.Random(42).shuffle(pieces)
    train_size = int(train_percent * len(pieces)) + 1

    train, test = [], []
    for i in range(train_size):
        for version in pieces[i]:
            train.append(version)

    for j in range(i + 1, len(pieces)):
        for version in pieces[j]:
            test.append(version)

    random.Random(42).shuffle(train)
    random.Random(42).shuffle(test)

    return train, test

def generate_shards(pieces, shards_amount=1, shard_prefix="", data_type="txt"):
    pieces_per_shard = int(len(pieces)/shards_amount)

    for i in range(shards_amount):
        if not os.path.exists("shards"):
            os.mkdir("shards")

        fp = open(os.path.join("shards", shard_prefix + "_shard_" + str(i) + ".txt"), "a")

        # If this data is midi, create a fake empty midi file because the dataloader requires it
        if data_type == "midi":
            fp_midi = open(os.path.join("shards", shard_prefix + "_shard_" + str(i) + ".midi"), "w")
            fp_midi.close()

        for j in range(pieces_per_shard):
            fp.write(pieces[i*pieces_per_shard + j])
            fp.write("\n ")

        fp.close()

pieces_path = sys.argv[1]
pieces = load_pieces(pieces_path)
train,test = split_data(pieces)

generate_shards(train, shards_amount=10, shard_prefix="train", data_type="midi")
generate_shards(test, shards_amount=1, shard_prefix="test", data_type="midi")
