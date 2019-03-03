# README

This is a object-oriented and self-contained reimplementation of OpenAI's Sentiment Neuron. It
currently supports text and midi sequences.

# Installing

$ pip install sentneuron

For running the interactive sampler, also install the following dependecies: Flask

$ pip install flask

# Running examples

This projects comes with two test scripts: one for trainning a new model and one for loading
pre-trainned models. Both scripts have an example of sampling sequences from the models.

To train a new model:
$ python examples/train_generative_txt.py

After trainning, this script stores the model inside the "trained_models/" folder. You can
then load it and sample sequences (either midi or text) using the following example:

To load a pre-trainned model:
$ python examples/test_generative_txt.py
