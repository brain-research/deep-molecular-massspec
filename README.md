# Deep learning for mass spectrometry for organic molecules.

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

## Quickstart:

TARGET_PATH_NAME=/tmp/massspec_predictions

To convert an sdf file into a TFRecord: 

```
python make_train_test_split.py \
--main_sdf_name=testdata/test_14_mend.sdf \
--replicates_sdf_name=testdata/test_2_mend.sdf \
--output_master_dir=$TARGET_PATH_NAME/spectra_tf_records \
--alsologtostderr
```

To train a model:

```
python molecule_estimator.py
--dataset_config_file=~/spectra_tf_records/query_replicates_val_predicted_replicates_val.json \
--train_steps=1000 \
--model_dir=$TARGET_PATH_NAME/models/output --hparams=make_spectra_plots=True \
--alsologtostderr
```

The aggregate training results will be logged to stdout. The final library
matching results can also be found in
$TARGET_PATH_NAME/models/output/predictions. These results report the query
inchikey, the matched inchikey, and the rank asigned to the true spectra.

It is also possible to view these results in tensorboard:

```
tensorboard --logdir=path/to/log-directory
```

To predict spectra for new data given a model, run:

```
make_predictions.py \
--input_file=testdata/test_14_record.gz \
--output_file=/tmp/models/output_predictions \
--model_checkpoint_path=/tmp/models/output/ \
--hparams=eval_batch_size=16 \
--alsologtostderr
```

## Datasets:

Most of our tasks use two datasets of spectra. One of these is the main library
(mainlib) which contains standardized spectra, e.g. the NIST 17 mass spectral
library. The other is a collection of spectra that might be found from a typical
experiment, e.g. the NIST replicates library.

Use make_train_test_split.py to make train/validation/test TFRecords. This
splits the mainlib and replicate sdf files into a train/validation/test split
according to the method specified by splitting_type. These splits are made
according to molecule, e.g. all spectra corresponding to one inchikey will be
placed in the same directory. The mainlib splits do not include any molecules
also included in the replicates datasets.

Two splitting_types are currently supported: 'random' and 'steroid'. Random
divides the molecules randomly. Steroid divides all the molecules containing the
4-ring steroid substructure into the validation/test sets, and places all other
molecules in the train sets.

For each of the dataset splits, the following files are generated:

-   *.tfrecord : a TFRecord containing information as listed in
    feature_map_constants.py
-   *.tfrecord.info: the number of record in the TFRecord file
-   *.inchikey.txt: A list of all of the inchikeys included in the TFRecord
    file.

A dataset config json then assigns each split for the spectral prediction and
library matching task. The library matching mask requires three sets of
datasets: observed, predicted, and query. The json assigns a list of the
datasets to each of these splits. Typically, the query are molecules that come
from the replicates dataset, and the observed spectra are from the main library.

## Library matching task:

One way of identifying a molecule is by finding the closest match of the
molecule's mass spectra in a library of standardized mass spectra. We duplicate
this setup in our library matching task.

This task uses three TFRecords, one for the query set, and two for the library
set. For the first library set, the experimental (observed) spectra will be used
directly. For the second set, our machine learning model will be used to predict
the spectra.

Several different similarity metrics are available for determining distances
between spectra. This includes the modified dot product proposed by Stein and
Scott (1994). A full list can be found in similarity.py

## Modeling methods:

All models make a prediction of the peaks at each mass in the mass spectra. Some
models used in this repo include MLP, and an RNN on the molecule's SMILES
string. The supported models can be found in molecule_predictors.py

## Glossary:

*inchikey* : a 27-character hash key for recording molecules.
https://en.wikipedia.org/wiki/International_Chemical_Identifier This repo uses
inchikeys as an identifying key for molecules; multiple entries in the dataset
that correspond to the same molecule are grouped together in dictionaries using
inchikeys as the key.

*sdf* : structure data file for molecules. Contains extra information about the
molecules as tags denoted by > <TAG_NAME>
https://en.wikipedia.org/wiki/Chemical_table_file#SDF

*smiles* : A string representation representing the structure of the molecule.
https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
