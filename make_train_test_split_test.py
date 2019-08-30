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

"""Tests for make_train_test_split.py and train_test_split_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

import dataset_setup_constants as ds_constants
import feature_map_constants as fmap_constants
import make_train_test_split
import mass_spec_constants as ms_constants
import parse_sdf_utils
import test_utils
import train_test_split_utils
import six
import tensorflow as tf


class MakeTrainTestSplitTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(MakeTrainTestSplitTest, self).setUp()
    test_data_directory = test_utils.test_dir('testdata/')
    self.temp_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    test_sdf_file_large = os.path.join(test_data_directory, 'test_14_mend.sdf')
    test_sdf_file_small = os.path.join(test_data_directory, 'test_2_mend.sdf')

    max_atoms = ms_constants.MAX_ATOMS
    self.mol_list_large = parse_sdf_utils.get_sdf_to_mol(
        test_sdf_file_large, max_atoms=max_atoms)
    self.mol_list_small = parse_sdf_utils.get_sdf_to_mol(
        test_sdf_file_small, max_atoms=max_atoms)
    self.inchikey_dict_large = train_test_split_utils.make_inchikey_dict(
        self.mol_list_large)
    self.inchikey_dict_small = train_test_split_utils.make_inchikey_dict(
        self.mol_list_small)
    self.inchikey_list_large = list(self.inchikey_dict_large.keys())
    self.inchikey_list_small = list(self.inchikey_dict_small.keys())

  def tearDown(self):
    tf.gfile.DeleteRecursively(self.temp_dir)
    super(MakeTrainTestSplitTest, self).tearDown()

  def encode(self, value):
    """Wrapper function for encoding strings in python 3."""
    return test_utils.encode(value, six.PY3)

  def test_all_lists_mutually_exclusive(self):
    list1 = ['1', '2', '3']
    list2 = ['2', '3', '4']
    try:
      train_test_split_utils.assert_all_lists_mutally_exclusive([list1, list2])
      raise ValueError('Sets with overlapping elements should have failed.')
    except ValueError:
      pass

  def test_make_inchikey_dict(self):
    self.assertLen(self.inchikey_dict_large, 11)
    self.assertLen(self.inchikey_dict_small, 2)

  def test_make_mol_list_from_inchikey_dict(self):
    mol_list = train_test_split_utils.make_mol_list_from_inchikey_dict(
        self.inchikey_dict_large, self.inchikey_list_large)
    self.assertCountEqual(mol_list, self.mol_list_large)

  def test_make_train_val_test_split_mol_lists(self):
    main_train_test_split = train_test_split_utils.TrainValTestFractions(
        0.5, 0.25, 0.25)

    inchikey_list_of_lists = (
        train_test_split_utils.make_train_val_test_split_inchikey_lists(
            self.inchikey_list_large, self.inchikey_dict_large,
            main_train_test_split))

    expected_lengths_of_inchikey_lists = [5, 2, 4]

    for expected_length, inchikey_list in zip(
        expected_lengths_of_inchikey_lists, inchikey_list_of_lists):
      self.assertLen(inchikey_list, expected_length)

    train_test_split_utils.assert_all_lists_mutally_exclusive(
        inchikey_list_of_lists)

    trunc_inchikey_list_large = self.inchikey_list_large[:6]
    inchikey_list_of_lists = [
        (train_test_split_utils.make_train_val_test_split_inchikey_lists(
            trunc_inchikey_list_large, self.inchikey_dict_large,
            main_train_test_split))
    ]

    expected_lengths_of_inchikey_lists = [3, 1, 2]
    for expected_length, inchikey_list in zip(
        expected_lengths_of_inchikey_lists, inchikey_list_of_lists):
      self.assertLen(inchikey_list, expected_length)

    train_test_split_utils.assert_all_lists_mutally_exclusive(
        inchikey_list_of_lists)

  def test_make_train_val_test_split_mol_lists_holdout(self):
    main_train_test_split = train_test_split_utils.TrainValTestFractions(
        0.5, 0.25, 0.25)
    holdout_inchikey_list_of_lists = (
        train_test_split_utils.make_train_val_test_split_inchikey_lists(
            self.inchikey_list_large,
            self.inchikey_dict_large,
            main_train_test_split,
            holdout_inchikey_list=self.inchikey_list_small))

    expected_lengths_of_inchikey_lists = [4, 2, 3]
    for expected_length, inchikey_list in zip(
        expected_lengths_of_inchikey_lists, holdout_inchikey_list_of_lists):
      self.assertLen(inchikey_list, expected_length)

    train_test_split_utils.assert_all_lists_mutally_exclusive(
        holdout_inchikey_list_of_lists)

  def test_make_train_val_test_split_mol_lists_family(self):
    train_test_split = train_test_split_utils.TrainValTestFractions(
        0.5, 0.25, 0.25)
    train_inchikeys, val_inchikeys, test_inchikeys = (
        train_test_split_utils.make_train_val_test_split_inchikey_lists(
            self.inchikey_list_large,
            self.inchikey_dict_large,
            train_test_split,
            holdout_inchikey_list=self.inchikey_list_small,
            splitting_type='diazo'))

    self.assertCountEqual(train_inchikeys, [
        'UFHFLCQGNIYNRP-UHFFFAOYSA-N', 'CCGKOQOJPYTBIH-UHFFFAOYSA-N',
        'ASTNYHRQIBTGNO-UHFFFAOYSA-N', 'UFHFLCQGNIYNRP-VVKOMZTBSA-N',
        'PVVBOXUQVSZBMK-UHFFFAOYSA-N'
    ])

    self.assertCountEqual(val_inchikeys + test_inchikeys, [
        'OWKPLCCVKXABQF-UHFFFAOYSA-N', 'COVPJOWITGLAKX-UHFFFAOYSA-N',
        'GKVDXUXIAHWQIK-UHFFFAOYSA-N', 'UCIXUAPVXAZYDQ-VMPITWQZSA-N'
    ])

    replicate_train_inchikeys, _, replicate_test_inchikeys = (
        train_test_split_utils.make_train_val_test_split_inchikey_lists(
            self.inchikey_list_small,
            self.inchikey_dict_small,
            train_test_split,
            splitting_type='diazo'))

    self.assertEqual(replicate_train_inchikeys[0],
                     'PNYUDNYAXSEACV-RVDMUPIBSA-N')
    self.assertEqual(replicate_test_inchikeys[0], 'YXHKONLOYHBTNS-UHFFFAOYSA-N')

  @parameterized.parameters('random', 'diazo')
  def test_make_train_test_split(self, splitting_type):
    """An integration test on a small dataset."""

    fpath = self.temp_dir

    # Create component datasets from two library files.
    main_train_val_test_fractions = (
        train_test_split_utils.TrainValTestFractions(0.5, 0.25, 0.25))
    replicates_val_test_fractions = (
        train_test_split_utils.TrainValTestFractions(0.0, 0.5, 0.5))

    (mainlib_inchikey_dict, replicates_inchikey_dict,
     component_inchikey_dict) = (
         make_train_test_split.make_mainlib_replicates_train_test_split(
             self.mol_list_large, self.mol_list_small, splitting_type,
             main_train_val_test_fractions, replicates_val_test_fractions))

    make_train_test_split.write_mainlib_split_datasets(
        component_inchikey_dict, mainlib_inchikey_dict, fpath,
        ms_constants.MAX_ATOMS, ms_constants.MAX_PEAK_LOC)

    make_train_test_split.write_replicates_split_datasets(
        component_inchikey_dict, replicates_inchikey_dict, fpath,
        ms_constants.MAX_ATOMS, ms_constants.MAX_PEAK_LOC)

    for experiment_setup in ds_constants.EXPERIMENT_SETUPS_LIST:
      # Create experiment json files
      tf.logging.info('Writing experiment setup for %s',
                      experiment_setup.json_name)
      make_train_test_split.check_experiment_setup(
          experiment_setup.experiment_setup_dataset_dict,
          component_inchikey_dict)

      make_train_test_split.write_json_for_experiment(experiment_setup, fpath)

      # Check that physical files for library matching contain all inchikeys
      dict_from_json = json.load(
          tf.gfile.Open(os.path.join(fpath, experiment_setup.json_name)))

      tf.logging.info(dict_from_json)
      library_files = (
          dict_from_json[ds_constants.LIBRARY_MATCHING_OBSERVED_KEY] +
          dict_from_json[ds_constants.LIBRARY_MATCHING_PREDICTED_KEY])
      library_files = [os.path.join(fpath, fname) for fname in library_files]

      hparams = tf.contrib.training.HParams(
          max_atoms=ms_constants.MAX_ATOMS,
          max_mass_spec_peak_loc=ms_constants.MAX_PEAK_LOC,
          intensity_power=1.0,
          batch_size=5)

      parse_sdf_utils.validate_spectra_array_contents(
          os.path.join(
              fpath,
              dict_from_json[ds_constants.SPECTRUM_PREDICTION_TRAIN_KEY][0]),
          hparams,
          os.path.join(fpath,
                       dict_from_json[ds_constants.TRAINING_SPECTRA_ARRAY_KEY]))

      dataset = parse_sdf_utils.get_dataset_from_record(
          library_files,
          hparams,
          mode=tf.estimator.ModeKeys.EVAL,
          all_data_in_one_batch=True)

      feature_names = [fmap_constants.INCHIKEY]
      label_names = [fmap_constants.ATOM_WEIGHTS]

      features, labels = parse_sdf_utils.make_features_and_labels(
          dataset, feature_names, label_names, mode=tf.estimator.ModeKeys.EVAL)

      with tf.Session() as sess:
        feature_values, _ = sess.run([features, labels])

      inchikeys_from_file = [
          ikey[0] for ikey in feature_values[fmap_constants.INCHIKEY].tolist()
      ]

      length_from_info_file = sum([
          parse_sdf_utils.parse_info_file(library_fname)['num_examples']
          for library_fname in library_files
      ])
      # Check that info file has the correct length for the file.
      self.assertLen(inchikeys_from_file, length_from_info_file)
      # Check that the TF.Record contains all of the inchikeys in our list.
      inchikey_list_large = [
          self.encode(ikey) for ikey in self.inchikey_list_large
      ]
      self.assertSetEqual(set(inchikeys_from_file), set(inchikey_list_large))


if __name__ == '__main__':
  tf.test.main()
