# Learning to Generate Music with Sentiment

This repository contains the source code to reproduce the ISMIR'19 paper "Learning to Generate Music with Sentiment", which proposes a deep learning method to generate music with a given sentiment (positive or negative).

## Installing

```
$ pip3 install sentneuron
```

## Results

The scripts to reproduce the results of the paper are all inside the examples/ directory:

0. Generate 3 train shards and 1 test shard to train the mLSTM generative model:
```
$ python3 examples/generate_shards.py -datadir input/generative/midi/vgmidi/ -data_type midi_perform -shards 3
```

1. Separate train and test shards in two different directories:

```
$ mkdir shards/test/
$ mv shards/test_shard_* shards/test/
```

2. Train a LSTM generative model using the unlabelled midi files:

```
python3 examples/train_generative.py -train_data input/generative/shards/ -test_data input/generative/midi/shards/test/test_shard_0.txt -data_type midi_perform
```

## Interactive

For running the interactive sampler, you will need to install the following dependecies: Flask

```
$ pip install flask
```

## TODO

- Improve generate_shard.py script to load any type of data (not only text)
- Update this README file with:
  - Tutorial to reproduce results of ISMIR paper.
  - Links to VGMIDI data.
