# Deep learning for Electron Ionization mass spectrometry for organic molecules(https://github.com/brain-research/deep-molecular-massspec/blob/master/toc.jpeg?raw=true)

This repository accompanies

Rapid Prediction of Electron–Ionization Mass Spectrometry Using Neural Networks\
Jennifer N. Wei, David Belanger, Ryan P. Adams, and D. Sculley\
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

## Required packages:

-   RDKit
-   Tensorflow

## Quickstart Guide for making model predictions

1. Create a directory and download the weights for the model.

```
$ MODEL_WEIGHTS_DIR=/tmp/neims_model
$ pushd $MODEL_WEIGHTS_DIR
$ wget https://storage.googleapis.com/deep-molecular-massspec/massspec_weights/massspec_weights.zip
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
