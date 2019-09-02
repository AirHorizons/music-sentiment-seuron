# LEARNING TO GENERATE MUSIC WITH SENTIMENT

This repository contains the source code to reproduce the ISMIR'19 paper "LEARNING TO GENERATE MUSIC WITH SENTIMENT", which proposes a deep learning method to generate music with a given sentiment (positive or negative).

## Installing

```
$ pip install sentneuron
```

## Examples

This projects comes with two test scripts: one for training a new model and one for loading
pre-trained models. Both scripts have an example of sampling sequences from the models.

To train a new model:

```
$ python examples/train_generative_txt.py
```

After training, this script stores the model inside the "trained_models/" folder. You can
then load it and sample sequences (either midi or text) using the following example:

To load a pre-trainned model:

```
$ python examples/load_generative.py
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

