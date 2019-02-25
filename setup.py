import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentneuron",
    version="0.1",
    author="Lucas N. Ferreira",
    author_email="lferreira@ucsc.edu",
    description="A object-oriented and self-contained simple implementation of OpenAI's Sentiment Neuron.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lucasnfe/Sentiment-Neuron",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
