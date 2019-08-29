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
r"""Train and evaluate massspec model.

Example usage:
molecule_estimator.py --train_steps=1000 --model_dir='/tmp/models/output' \
--dataset_config_file=testdata/test_dataset_config_file.json --alsologtostderr
"""

from __future__ import print_function
import json
import os

from absl import flags
import dataset_setup_constants as ds_constants
import feature_map_constants as fmap_constants
import library_matching
import molecule_predictors
import parse_sdf_utils
import six
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_config_file', None,
    'JSON file specifying the various filenames necessary for training and '
    'evaluation. See make_input_fn() for more details.')
flags.DEFINE_string(
    'hparams', '', 'Hyperparameter values to override the defaults.'
    'Format: params1=value1,params2=value2, ...'
    'Possible parameters: max_atoms, max_mass_spec_peak_loc,'
    'batch_size, epochs, do_linear_regression,'
    'get_mass_spec_features, init_weights, init_bias')
flags.DEFINE_integer('train_steps', None,
                     'The number of steps to run training for.')
flags.DEFINE_integer(
    'train_steps_per_iteration', 1000,
    'how frequently to evaluate (only used when schedule =='
    ' continuous_train_and_eval')

flags.DEFINE_string('model_dir', '',
                    'output directory for checkpoints and events files')
flags.DEFINE_string('warm_start_dir', None,
                    'directory to warm start model from')

flags.DEFINE_enum('model_type', 'mlp',
                  molecule_predictors.MODEL_REGISTRY.keys(),
                  'Type of model to use.')
OUTPUT_HPARAMS_CONFIG_FILE_BASE = 'command_line_arguments.txt'


def make_input_fn(dataset_config_file,
                  hparams,
                  mode,
                  features_to_load,
                  load_library_matching_data,
                  data_dir=None):
  """Make input functions returning features and labels.

  In our downstream code, it is advantageous to put both features
  and labels together in the same nested structure. However, tf.estimator
  requires input_fn() to return features and labels, so here our input_fn
  returns dummy labels that will not be used.

  Args:
    dataset_config_file: filename of JSON file containing a dict mapping dataset
      names to data files. The required keys are:
      'SPECTRUM_PREDICTION_TRAIN': training data
      'SPECTRUM_PREDICTION_TEST': eval data on which we evaluate the same loss
        function that is used for training
      'LIBRARY_MATCHING_OBSERVED': data for library matching where we use
        ground-truth spectra
      'LIBRARY_MATCHING_PREDICTED': data for library matching where we use
        predicted spectra
      'LIBRARY_MATCHING_QUERY': data with observed spectra used for queries in
        the library
      matching task

      For each data file with name <fname>, we read high-level information about
      the data from a separate file with the name <fname>.info. See
      parse_sdf_utils.parse_info_file() for the expected format of that file.

    hparams: hparams required for parsing; includes features such as max_atoms,
              max_mass_spec_peak_loc, and batch_size
    mode: Set whether training or test mode
    features_to_load: list of keys to load from the input data
    load_library_matching_data: whether to load library matching data.
    data_dir: The directory containing the file names referred to in
      dataset_config_file. If None (the default) then this is assumed to be the
      directory containing dataset_config_file.
  Returns:
    A function for creating features and labels from a dataset.
  """
  with tf.gfile.Open(dataset_config_file, 'r') as f:
    filenames = json.load(f)

  if data_dir is None:
    data_dir = os.path.dirname(dataset_config_file)

  def _input_fn(record_fnames,
                all_data_in_one_batch,
                load_training_spectrum_library=False):
    """Reads TFRecord from a list of record file names."""
    if not record_fnames:
      return None

    record_fnames = [os.path.join(data_dir, r_name) for r_name in record_fnames]
    dataset = parse_sdf_utils.get_dataset_from_record(
        record_fnames,
        hparams,
        mode=mode,
        features_to_load=(features_to_load + hparams.label_names),
        all_data_in_one_batch=all_data_in_one_batch)
    dict_to_return = parse_sdf_utils.make_features_and_labels(
        dataset, features_to_load, hparams.label_names, mode=mode)[0]

    if load_training_spectrum_library:
      library_file = os.path.join(
          '/readahead/128M/',
          filenames[ds_constants.TRAINING_SPECTRA_ARRAY_KEY])
      train_library = parse_sdf_utils.load_training_spectra_array(library_file)
      train_library = tf.convert_to_tensor(train_library, dtype=tf.float32)

      dict_to_return['SPECTRUM_PREDICTION_LIBRARY'] = train_library

    return dict_to_return

  load_training_spectrum_library = hparams.loss == 'max_margin'

  if load_library_matching_data:

    def _wrapped_input_fn():
      """Construct data for various eval tasks."""

      data_to_return = {
          fmap_constants.SPECTRUM_PREDICTION:
              _input_fn(
                  filenames[ds_constants.SPECTRUM_PREDICTION_TEST_KEY],
                  all_data_in_one_batch=False,
                  load_training_spectrum_library=load_training_spectrum_library)
      }

      if hparams.do_library_matching:
        observed = _input_fn(
            filenames[ds_constants.LIBRARY_MATCHING_OBSERVED_KEY],
            all_data_in_one_batch=True)
        predicted = _input_fn(
            filenames[ds_constants.LIBRARY_MATCHING_PREDICTED_KEY],
            all_data_in_one_batch=True)
        query = _input_fn(
            filenames[ds_constants.LIBRARY_MATCHING_QUERY_KEY],
            all_data_in_one_batch=False)
        data_to_return[fmap_constants.
                       LIBRARY_MATCHING] = library_matching.LibraryMatchingData(
                           observed=observed, predicted=predicted, query=query)

      return data_to_return, tf.constant(0)
  else:

    def _wrapped_input_fn():
      # See the above comment about why we use dummy labels.
      return {
          fmap_constants.SPECTRUM_PREDICTION:
              _input_fn(
                  filenames[ds_constants.SPECTRUM_PREDICTION_TRAIN_KEY],
                  all_data_in_one_batch=False,
                  load_training_spectrum_library=load_training_spectrum_library)
      }, tf.constant(0)

  return _wrapped_input_fn


def _log_command_line_string(model_type, model_dir, hparams):
  """Log command line args to replicate hparam configuration."""

  config_string = '--model_type=%s ' % (model_type)

  # Note that the rendered string will not be able to be parsed using
  # hparams.parse() if any of the hparam values have commas or '=' signs.
  hparams_string = ','.join(
      ['%s=%s' % (key, value) for key, value in six.iteritems(
          hparams.values())])

  config_string += ' --hparams=%s\n' % hparams_string
  output_file = os.path.join(model_dir, OUTPUT_HPARAMS_CONFIG_FILE_BASE)
  tf.gfile.MakeDirs(model_dir)
  tf.logging.info('Writing command line config string to %s.' % output_file)

  with tf.gfile.Open(output_file, 'w') as f:
    f.write(config_string)


def make_model_fn(prediction_helper, dataset_config_file, model_dir):
  """Returns a model function for estimator given prediction base class.

  Args:
    prediction_helper : Helper class containing prediction, loss, and evaluation
                        metrics
    dataset_config_file: see make_input_fn.
    model_dir : directory for writing output files. If model dir is not None,
    a file containing all of the necessary command line flags to rehydrate
    the model will be written in model_dir.
  Returns:
    A function that returns a tf.EstimatorSpec
  """

  def _model_fn(features, labels, params, mode=None):
    """Returns tf.EstimatorSpec."""

    # Input labels are ignored. All data are in features.
    del labels

    if model_dir is not None:
      _log_command_line_string(prediction_helper.model_type, model_dir, params)

    pred_op, pred_op_for_loss = prediction_helper.make_prediction_ops(
        features[fmap_constants.SPECTRUM_PREDICTION], params, mode)
    loss_op = prediction_helper.make_loss(
        pred_op_for_loss, features[fmap_constants.SPECTRUM_PREDICTION], params)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss=loss_op,
          global_step=tf.train.get_global_step(),
          clip_gradients=params.gradient_clip,
          learning_rate=params.learning_rate,
          optimizer='Adam')
      eval_op = None
    elif mode == tf.estimator.ModeKeys.PREDICT:
      train_op = None
      eval_op = None
    elif mode == tf.estimator.ModeKeys.EVAL:
      train_op = None
      eval_op = prediction_helper.make_evaluation_metrics(
          features, params, dataset_config_file, output_dir=model_dir)
    else:
      raise ValueError('Invalid mode. Must be '
                       'tf.estimator.ModeKeys.{TRAIN,PREDICT,EVAL}.')
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_op,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=eval_op)

  return _model_fn


def make_estimator_and_inputs(run_config,
                              hparams,
                              prediction_helper,
                              dataset_config_file,
                              train_steps,
                              model_dir,
                              warm_start_dir=None):
  """Make Estimator-compatible Estimator and input_fn for train and eval."""

  model_fn = make_model_fn(prediction_helper, dataset_config_file, model_dir)

  train_input_fn = make_input_fn(
      dataset_config_file=dataset_config_file,
      hparams=hparams,
      mode=tf.estimator.ModeKeys.TRAIN,
      features_to_load=prediction_helper.features_to_load(hparams),
      load_library_matching_data=False)

  eval_input_fn = make_input_fn(
      dataset_config_file=dataset_config_file,
      hparams=hparams,
      mode=tf.estimator.ModeKeys.EVAL,
      features_to_load=prediction_helper.features_to_load(hparams),
      load_library_matching_data=True)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      params=hparams,
      config=run_config,
      warm_start_from=warm_start_dir)

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=None)

  return estimator, train_spec, eval_spec


def main(_):
  prediction_helper = molecule_predictors.get_prediction_helper(
      FLAGS.model_type)

  hparams = prediction_helper.get_default_hparams()
  hparams.parse(FLAGS.hparams)

  config = tf.contrib.learn.RunConfig(model_dir=FLAGS.model_dir)

  (estimator, train_spec, eval_spec) = make_estimator_and_inputs(
      config, hparams, prediction_helper, FLAGS.dataset_config_file,
      FLAGS.train_steps, FLAGS.model_dir, FLAGS.warm_start_dir)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  tf.app.run(main)
