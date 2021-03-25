# Deep learning for Electron Ionization mass spectrometry for organic molecules
![TOC](https://github.com/brain-research/deep-molecular-massspec/blob/master/neims_toc.jpeg?raw=true)

This repository accompanies

[Rapid Prediction of Electron–Ionization Mass Spectrometry Using Neural Networks](https://pubs.acs.org/doi/10.1021/acscentsci.9b00085)\
[Jennifer N. Wei](https://ai.google/research/people/JenniferNWei),
[David Belanger](https://davidbelanger.github.io/), [Ryan P. Adams](https://www.cs.princeton.edu/~rpa/),
and [D. Sculley](https://www.eecs.tufts.edu/~dsculley/)\
ACS Central Science 2019 5 (4), 700-708\
DOI: 10.1021/acscentsci.9b00085


## Introduction

We predict the mass spectrometry spectra of molecules using deep learning
techniques applied to various molecule representations. The performance behavior
is evaluated with a custom-made library matching task. In this task we identify
molecules by matching its spectra to a library of labeled spectra. As a
baseline, this library contains all of the molecules in the NIST main library,
which mimics the behavior currently used by experimental chemists. To test our
predictions, we replace portions of the library with spectra predictions from
our model. This task is described in more detail below.

## Required Packages:

It is recommended to use [Anaconda](https://www.anaconda.com/distribution/) with a Python 3.6 environment to install these packages.
-   [RDKit](https://www.rdkit.org/docs/Install.html)
-   [Tensorflow](https://www.tensorflow.org/install) 

Most of the packages required here can be installed with conda install tensorflow=1.13 rdkit matplotlib and pip install absl-py.

## Quickstart Guide for Making Model Predictions

1. Create a directory and download the weights for the model.

```
$ MODEL_WEIGHTS_DIR=/home/path/to/model
$ mkdir $MODEL_WEIGHTS_DIR
$ pushd $MODEL_WEIGHTS_DIR
$ curl -o https://storage.googleapis.com/deep-molecular-massspec/massspec_weights/massspec_weights.zip
$ unzip massspec_weights.zip
$ popd
```

2. Run the model prediction on the example molecule

```
$ python make_spectra_prediction.py \
--input_file=examples/pentachlorobenzene.sdf \
--output_file=/tmp/annotated.sdf \
--weights_dir=$MODEL_WEIGHTS_DIR/massspec_weights
```

## Training splits for benchmarking purposes
The molecules used for the training, validation, and test sets can be found under the
directory *training_splits*. The molecules are provided in
inchikey and smiles format.


## To cite this work:

@article{doi:10.1021/acscentsci.9b00085,\
author = {Wei, Jennifer N. and Belanger, David and Adams, Ryan P. and Sculley, D.},\
title = {Rapid Prediction of Electron–Ionization Mass Spectrometry Using Neural Networks},\
journal = {ACS Central Science},\
volume = {5},\
number = {4},\
pages = {700-708},\
year = {2019},\
doi = {10.1021/acscentsci.9b00085},\
URL = {https://doi.org/10.1021/acscentsci.9b00085},\
}
