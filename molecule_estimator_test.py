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

"""Tests for molecule_estimator."""

from __future__ import print_function
import json
import os
import tempfile


from absl.testing import absltest
from absl.testing import parameterized

import dataset_setup_constants as ds_constants
import mass_spec_constants as ms_constants
import molecule_estimator
import molecule_predictors
import parse_sdf_utils
import plot_spectra_utils
import test_utils
import numpy as np
import tensorflow as tf


class MoleculeEstimatorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Sets up a dataset json for regular, baseline, and all_predicted cases."""
    super(MoleculeEstimatorTest, self).setUp()
    self.test_data_directory = test_utils.test_dir('testdata/')
    record_file = os.path.join(self.test_data_directory, 'test_14_record.gz')

    self.num_eval_examples = parse_sdf_utils.parse_info_file(record_file)[
        'num_examples']
    self.temp_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self.default_dataset_config_file = os.path.join(self.temp_dir,
                                                    'dataset_config.json')
    self.baseline_dataset_config_file = os.path.join(
        self.temp_dir, 'baseline_dataset_config.json')
    self.all_predicted_dataset_config_file = os.path.join(
        self.temp_dir, 'all_predicted_dataset_config.json')

    dataset_names = [
        ds_constants.SPECTRUM_PREDICTION_TRAIN_KEY,
        ds_constants.SPECTRUM_PREDICTION_TEST_KEY,
        ds_constants.LIBRARY_MATCHING_OBSERVED_KEY,
        ds_constants.LIBRARY_MATCHING_PREDICTED_KEY,
        ds_constants.LIBRARY_MATCHING_QUERY_KEY
    ]

    default_dataset_config = {key: [record_file] for key in dataset_names}
    default_dataset_config[
        ds_constants.TRAINING_SPECTRA_ARRAY_KEY] = os.path.join(
            self.test_data_directory, 'test_14.spectra_library.npy')
    with tf.gfile.Open(self.default_dataset_config_file, 'w') as f:
      json.dump(default_dataset_config, f)

    # Test estimator behavior when predicted set is empty
    baseline_dataset_config = dict(
        [(key, [record_file])
         if key != ds_constants.LIBRARY_MATCHING_PREDICTED_KEY else (key, [])
         for key in dataset_names])
    baseline_dataset_config[
        ds_constants.TRAINING_SPECTRA_ARRAY_KEY] = os.path.join(
            self.test_data_directory, 'test_14.spectra_library.npy')
    with tf.gfile.Open(self.baseline_dataset_config_file, 'w') as f:
      json.dump(baseline_dataset_config, f)

    # Test estimator behavior when observed set is empty
    all_predicted_dataset_config = dict(
        [(key, [record_file])
         if key != ds_constants.LIBRARY_MATCHING_OBSERVED_KEY else (key, [])
         for key in dataset_names])
    all_predicted_dataset_config[
        ds_constants.TRAINING_SPECTRA_ARRAY_KEY] = os.path.join(
            self.test_data_directory, 'test_14.spectra_library.npy')
    with tf.gfile.Open(self.all_predicted_dataset_config_file, 'w') as f:
      json.dump(all_predicted_dataset_config, f)

  def tearDown(self):
    tf.gfile.DeleteRecursively(self.temp_dir)
    super(MoleculeEstimatorTest, self).tearDown()

  def _get_loss_history(self, checkpoint_dir):
    """Get list of train losses from events file."""
    losses = []
    for event_file in tf.gfile.Glob(
        os.path.join(checkpoint_dir, 'events.out.tfevents.*')):
      for event in tf.train.summary_iterator(event_file):
        for v in event.summary.value:
          if v.tag == 'loss':
            losses.append(v.simple_value)
    return losses

  def _run_estimator(self, prediction_helper, get_hparams, dataset_config_file):
    """Helper function for running molecule_estimator."""
    checkpoint_dir = self.temp_dir
    config = tf.contrib.learn.RunConfig(
        model_dir=checkpoint_dir, save_summary_steps=1)
    (estimator, train_spec,
     eval_spec) = molecule_estimator.make_estimator_and_inputs(
         config,
         get_hparams(),
         prediction_helper,
         dataset_config_file,
         train_steps=10,
         model_dir=self.temp_dir)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    loss_history = self._get_loss_history(checkpoint_dir)

    init_loss = loss_history[0]
    loss = loss_history[-1]
    if not np.isfinite(loss):
      raise ValueError('Final loss is not finite: %f' % loss)

    tf.logging.info('initial loss : {} final loss : {}'.format(init_loss, loss))

    self.assertNotEqual(loss, init_loss,
                        ('Loss did not change after brief testing: '
                         'init = %f, final = %f.') % (init_loss, loss))

  @parameterized.parameters(
      ('linear', 0, 'generalized_mse', False),
      ('mlp', 1, 'generalized_mse', True, True),
      ('linear', 0, 'cross_entropy', True),
      ('mlp', 2, 'generalized_mse', False),
      ('linear', 0, 'max_margin', True),
      ('smiles_rnn', 0, 'generalized_mse', True, True))
  def test_run_estimator(self, model_type, num_hidden_layers, loss_type,
                         do_library_matching, bidirectional_prediction=False):
    """Integration test for molecule_estimator."""
    prediction_helper = molecule_predictors.get_prediction_helper(model_type)

    def get_hparams():
      hparams = prediction_helper.get_default_hparams()
      hparams.set_hparam('loss', loss_type)
      hparams.set_hparam('do_library_matching', do_library_matching)
      hparams.set_hparam('bidirectional_prediction', bidirectional_prediction)

      # To test batching and padding in library matching, set the
      # eval_batch_size such that it does not divide the number of examples
      # in the test set.
      eval_batch_size = np.int32(np.floor(self.num_eval_examples / 2) - 1)
      assert eval_batch_size > 0, ('The evaluation data is not big enough to '
                                   'support using multiple batches, where the '
                                   'batch size does not divide the total '
                                   'number of examples.')
      hparams.set_hparam('eval_batch_size', eval_batch_size)

      if model_type == 'mlp':
        hparams.set_hparam('num_hidden_layers', num_hidden_layers)
      return hparams

    self._run_estimator(prediction_helper, get_hparams,
                        self.default_dataset_config_file)

  def test_run_estimator_on_baseline(self):
    prediction_helper = molecule_predictors.get_prediction_helper('baseline')
    self._run_estimator(prediction_helper,
                        prediction_helper.get_default_hparams,
                        self.baseline_dataset_config_file)

  def test_run_estimator_on_all_predicted(self):
    prediction_helper = molecule_predictors.get_prediction_helper('mlp')
    self._run_estimator(prediction_helper,
                        prediction_helper.get_default_hparams,
                        self.all_predicted_dataset_config_file)

  def test_plot_true_and_predicted_spectra(self):
    """Test if plot is successfully generated given two spectra."""
    max_mass_spec_peak_loc = ms_constants.MAX_PEAK_LOC
    true_spectra = np.zeros(max_mass_spec_peak_loc)
    predicted_spectra = np.zeros(max_mass_spec_peak_loc)
    true_spectra[3:6] = 60
    predicted_spectra[300] = 999
    true_spectra[200] = 780

    test_figure_path_name = os.path.join(self.temp_dir, 'test.png')
    generated_plot = plot_spectra_utils.plot_true_and_predicted_spectra(
        true_spectra, predicted_spectra, output_filename=test_figure_path_name)

    self.assertEqual(
        np.shape(generated_plot),
        plot_spectra_utils.SPECTRA_PLOT_DIMENSIONS_RGB)
    self.assertTrue(os.path.exists(test_figure_path_name))


if __name__ == '__main__':
  tf.test.main()
