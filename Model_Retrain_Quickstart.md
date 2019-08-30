## Quickstart 

The following is a tutorial for training a new model.

Setup a directory for predictions:

```
$ TARGET_PATH_NAME=/tmp/massspec_predictions
```

To convert an sdf file into a TFRecord and make the training and test splits: 

```
$ python make_train_test_split.py --main_sdf_name=testdata/test_14_mend.sdf
--replicates_sdf_name=testdata/test_2_mend.sdf \\ \
--output_master_dir=$TARGET_PATH_NAME/spectra_tf_records
```

To train a model:

```
python molecule_estimator.py
--dataset_config_file=~/spectra_tf_records/query_replicates_val_predicted_replicates_val.json
--train_steps=1000 \\ \
--model_dir=$TARGET_PATH_NAME/models/output --hparams=make_spectra_plots=True
--alsologtostderr
```

The aggregate training results will be logged to stdout. The final library
matching results can also be found in
$TARGET_PATH_NAME/models/output/predictions. These results report the query
inchikey, the matched inchikey, and the rank asigned to the true spectra.

It is also possible to view these results in tensorboard: \
tensorboard --logdir=path/to/log-directory

To predict spectra for new data given a model, run:

```
python make_predictions_from_tfrecord.py \
--input_file=testdata/test_14_record.gz \
--output_file=/tmp/models/output_predictions \
--model_checkpoint_path=/tmp/models/output/ \
--hparams=eval_batch_size=16 \
--alsologtostderr
```
