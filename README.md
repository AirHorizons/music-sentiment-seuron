# Learning to Generate Music with Sentiment

This repository contains the source code to reproduce the ISMIR'19 paper "Learning to Generate Music with Sentiment", which proposes a deep learning method to generate music with a given sentiment (positive or negative).

## Installing

```
$ pip3 install sentneuron
```

## Results

The scripts to reproduce the results of the paper are all inside the examples/ directory:

#### 0. Generate 3 train shards and 1 test shard to train the mLSTM generative model:
```
$ python3 examples/generate_shards.py -datadir input/generative/midi/vgmidi/ -data_type midi_perform -shards 3
```

#### 1. Separate train and test shards in two different directories:

```
$ mkdir shards/test/
$ mv shards/test_shard_* shards/test/
```

#### 2. Move shards data to the input directory:

```
$ mv shards input/generative/midi/vgmidi-shards
```

#### 3. Train a LSTM generative model using the unlabelled midi files:

```
python3 examples/train_generative.py -train_data input/generative/midi/vgmidi-shards -test_data input/generative/midi/vgmidi-shards/test/test_shard_0.txt -data_type midi_perform -save_path trained/
```

#### 4. Train a Logistic Recression (LR) classifier using the trained (generative) LSTM hidden layer to encode the labelled midi files and evolve LR weights to generate positive pieces:

```
python3 examples/train_classifier_unsupervised.py -model_path trained/vgmidi-shards -sent_data_path input/classifier/midi/vgmidi/vgmidi.csv -results_path output/ -sentiment 1
```

#### 5. Train a Logistic Recression (LR) classifier using the trained (generative) LSTM hidden layer to encode the labelled midi files and evolve LR weights to generate negative pieces:

```
python3 examples/train_classifier_unsupervised.py -model_path trained/vgmidi-shards -sent_data_path input/classifier/midi/vgmidi/vgmidi.csv -results_path output/ -sentiment 0
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
