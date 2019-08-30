# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Run massspec model on data and write out predictions.

Example usage:
blaze-bin/third_party/py/deep_molecular_massspec/make_predictions \
--alsologtostderr --input_file=testdata/test_14_record.gz \
--output_file=/tmp/models/output_predictions \
--model_checkpoint_path=/tmp/models/output/ \
--hparams=eval_batch_size=16

This saves a numpy archive to FLAGS.output_file that contains a dictionary
where the keys are inchikeys and values are 1D np arrays for spectra.

You should load this dict downstream using:
data_dict = np.load(data_file).item()

(Note that .item() is necessary because np.load returns a 0-D array,
where the first element is the desired dictionary.)
"""

from __future__ import print_function
import json
import os
import tempfile


from absl import flags

import dataset_setup_constants as ds_constants
import feature_map_constants as fmap_constants

# Note that many FLAGS are inherited from molecule_estimator
import molecule_estimator

import molecule_predictors
import plot_spectra_utils
import util

import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None, 'Input TFRecord file or a '
    'globble file pattern for TFRecord files')
flags.DEFINE_string(
    'model_checkpoint_path', None,
    'Path to model checkpoint. If a directory, the most '
    'recent model checkpoint in this directory will be used. If a file, it '
    'should be of the form /.../name-of-the-file.ckpt-10000')
flags.DEFINE_bool(
    'save_spectra_plots', True,
    'Make plots of true and predicted spectra for each query molecule.')
flags.DEFINE_string('output_file', None,
                    'Location where outputs will be written.')


def _make_features_and_labels_from_tfrecord(input_file_pattern, hparams,
                                            features_to_load):
  """Construct features and labels Tensors to be consumed by model_fn."""

  def _make_tmp_dataset_config_file(input_filenames):
    """Construct a temporary config file that points to input_filename."""

    _, tmp_file = tempfile.mkstemp()
    dataset_config = {
        ds_constants.SPECTRUM_PREDICTION_TRAIN_KEY: input_filenames
    }

    # TODO(b/135189673): Replace this with tf.gfile.Open once the type
    #   issue is fixed.
    with open(tmp_file, 'w') as f:
      json.dump(dataset_config, f)

    return tmp_file

  input_files = tf.gfile.Glob(input_file_pattern)
  if not input_files:
    raise ValueError('No files found matching %s' % input_file_pattern)

  data_dir, _ = os.path.split(input_files[0])
  data_basenames = [os.path.split(filename)[1] for filename in input_files]
  dataset_config_file = _make_tmp_dataset_config_file(data_basenames)

  mode = tf.estimator.ModeKeys.PREDICT
  input_fn = molecule_estimator.make_input_fn(
      dataset_config_file=dataset_config_file,
      hparams=hparams,
      mode=mode,
      features_to_load=features_to_load,
      data_dir=data_dir,
      load_library_matching_data=False)
  tf.gfile.Remove(dataset_config_file)
  return input_fn()


def _make_features_labels_and_estimator(model_type, hparam_string, input_file):
  """Construct input ops and EstimatorSpec for massspec model."""

  prediction_helper = molecule_predictors.get_prediction_helper(model_type)
  hparams = prediction_helper.get_default_hparams()
  hparams.parse(hparam_string)

  model_fn = molecule_estimator.make_model_fn(
      prediction_helper, dataset_config_file=None, model_dir=None)

  features_to_load = prediction_helper.features_to_load(hparams)
  features, labels = _make_features_and_labels_from_tfrecord(
      input_file, hparams, features_to_load)

  estimator_spec = model_fn(
      features, labels, hparams, mode=tf.estimator.ModeKeys.PREDICT)

  return features, labels, estimator_spec


def _save_plot_figure(key, prediction, true_spectrum, results_dir):
  """A helper function that makes and saves plots of true and predicted spectra."""
  spectra_plot_file_name = plot_spectra_utils.name_plot_file(
      plot_spectra_utils.PlotModeKeys.PREDICTED_SPECTRUM, key, file_type='png')

  # Rescale the true/predicted spectra
  true_spectrum = (
      true_spectrum / true_spectrum.max() *
      plot_spectra_utils.MAX_VALUE_OF_TRUE_SPECTRA)
  prediction = (
      prediction / prediction.max() *
      plot_spectra_utils.MAX_VALUE_OF_TRUE_SPECTRA)

  plot_spectra_utils.plot_true_and_predicted_spectra(
      true_spectrum,
      prediction,
      output_filename=os.path.join(results_dir, spectra_plot_file_name),
      rescale_mz_axis=True)


def main(_):

  features, labels, estimator_spec = _make_features_labels_and_estimator(
      FLAGS.model_type, FLAGS.hparams, FLAGS.input_file)
  del labels  # Unused

  pred_op = estimator_spec.predictions
  inchikey_op = features[fmap_constants.SPECTRUM_PREDICTION][
      fmap_constants.INCHIKEY]
  ops_to_fetch = [inchikey_op, pred_op]
  if FLAGS.save_spectra_plots:
    true_spectra_op = features[fmap_constants.SPECTRUM_PREDICTION][
        fmap_constants.DENSE_MASS_SPEC]
    ops_to_fetch.append(true_spectra_op)

  results = {}
  results_dir = os.path.dirname(FLAGS.output_file)
  tf.gfile.MakeDirs(results_dir)

  def process_fetched_values_fn(fetched_values):
    """Processes output values from estimator."""
    if FLAGS.save_spectra_plots:
      keys, predictions, true_spectra = fetched_values
      for key, prediction, true_spectrum in zip(keys, predictions,
                                                true_spectra):
        # Dereference the singleton np string array to get the actual string.
        key = key[0]
        results[key] = prediction
        # Spectra plots are saved to file.
        _save_plot_figure(key, prediction, true_spectrum, results_dir)
    else:
      keys, predictions = fetched_values
      for key, prediction in zip(keys, predictions):
        # Dereference the singleton np string array to get the actual string.
        key = key[0]
        results[key] = prediction

  util.run_graph_and_process_results(ops_to_fetch, FLAGS.model_checkpoint_path,
                                     process_fetched_values_fn)

  tf.gfile.MakeDirs(os.path.dirname(FLAGS.output_file))
  np.save(FLAGS.output_file, results)


if __name__ == '__main__':
  for flag in ['input_file', 'model_checkpoint_path', 'output_file']:
    flags.mark_flag_as_required(flag)

  tf.app.run(main)
