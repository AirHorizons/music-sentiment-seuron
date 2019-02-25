# README

This is a object-oriented and self-contained reimplementation of OpenAI's Sentiment Neuron. It
currently supports text and midi sequences.

# Installing Dependecies

The core features of this projects has a few dependecies: NumPy, PyTorch, Sklearn, Music21

$ pip install numpy torch torchvision sklearn music21

For running the interactive sampler, also install the following dependecies: pygame

$ pip install pygame

# Running examples

This projects comes with two test scripts: one for trainning a new model and one for loading
pre-trainned models. Both scripts have an example of sampling sequences from the models.

To train a new model:
$ python test_train.py

To load a pre-trainned model:
$ python test_load.py
