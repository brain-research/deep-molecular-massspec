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

"""Tests for parse_sdf_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile


from absl.testing import absltest

import feature_map_constants as fmap_constants
import feature_utils
import mass_spec_constants as ms_constants
import parse_sdf_utils
import test_utils
import numpy as np
from rdkit import Chem
import six
import tensorflow as tf


class ParseSdfUtilsTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(ParseSdfUtilsTest, self).setUp()
    self.test_data_directory = test_utils.test_dir('testdata/')
    self.test_file_long = os.path.join(self.test_data_directory,
                                       'test_14_mend.sdf')
    self.test_file_short = os.path.join(self.test_data_directory,
                                        'test_2_mend.sdf')
    self.temp_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())

    # Expected result for list of molecule dicts
    self.expected_mol_dicts = [{
        fmap_constants.NAME: 'Methane, diazo-',
        fmap_constants.INCHIKEY: 'YXHKONLOYHBTNS-UHFFFAOYSA-N',
        fmap_constants.MOLECULAR_FORMULA: 'CH2N2',
        fmap_constants.SMILES: 'C=[N+]=[N-]',
        'parsed_smiles': [28, 18, 81, 51, 4, 83, 18, 81, 51, 5, 83],
        fmap_constants.SMILES_TOKEN_LIST_LENGTH: 11
    }, {
        fmap_constants.NAME: (
            '(4-(4-Chlorphenyl)-3-morpholino-pyrrol-2-yl)-butenedioic acid,'
            ' dimethyl ester'),
        fmap_constants.INCHIKEY:
            'PNYUDNYAXSEACV-RVDMUPIBSA-N',
        fmap_constants.MOLECULAR_FORMULA:
            'C20H21ClN2O5',
        fmap_constants.SMILES:
            'COC(=O)/C=C(/C(=O)OC)c1[nH]cc(-c2ccc(Cl)cc2)c1N1CCOCC1',
        'parsed_smiles': [
            28, 55, 28, 2, 18, 55, 3, 7, 28, 18, 28, 2, 7, 28, 2, 18, 55, 3, 55,
            28, 3, 84, 9, 81, 85, 40, 83, 84, 84, 2, 5, 84, 10, 84, 84, 84, 2,
            31, 3, 84, 84, 10, 3, 84, 9, 51, 9, 28, 28, 55, 28, 28, 9
        ],
        fmap_constants.SMILES_TOKEN_LIST_LENGTH:
            53,
    }]
    for mol_dict in self.expected_mol_dicts:
      token_arr = mol_dict['parsed_smiles']
      sequence_length = mol_dict[
          fmap_constants.SMILES_TOKEN_LIST_LENGTH]
      mol_dict['parsed_smiles'] = np.pad(
          token_arr, (0, ms_constants.MAX_TOKEN_LIST_LENGTH - sequence_length),
          'constant')

    mol_weights = [42.0217981, 404.1139]
    atom_weights_list = [[12.011, 14.007, 14.007], [
        12.011, 15.999, 12.011, 15.999, 12.011, 12.011, 12.011, 15.999, 15.999,
        12.011, 12.011, 14.007, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011,
        35.453, 12.011, 12.011, 12.011, 14.007, 12.011, 12.011, 15.999, 12.011,
        12.011
    ]]
    atom_ids_list = [[6, 7, 7], [
        6, 8, 6, 8, 6, 6, 6, 8, 8, 6, 6, 7, 6, 6, 6, 6, 6, 6, 17, 6, 6, 6, 7, 6,
        6, 8, 6, 6
    ]]
    adjacency_matrix_list = [
        np.array(
            [
                0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
                2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0.
            ],
            dtype='int32'),
        np.array(
            [
                0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 2.,
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1., 0., 0., 0.,
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 2., 1., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 4., 0., 1., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 1., 0., 4., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4.,
                0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0.,
                1., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0., 4., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 4., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                4., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0.
            ],
            dtype='int32')
    ]
    mass_spec_peak_locs = [[22, 23, 24, 25, 26, 27, 28, 30, 31, 32], [
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50,
        51
    ]]
    mass_spec_peak_intensities = [[
        110, 220, 999, 25, 12, 58, 179, 22, 110, 425
    ], [
        12, 7, 28, 999, 57, 302, 975, 8, 53, 176, 99, 122, 117, 155, 9, 7, 6,
        28, 59
    ]]

    # Special hparams so that expected arrays can be smaller.
    self.hparams = tf.contrib.training.HParams(
        intensity_power=1.0,
        max_atoms=30,
        max_mass_spec_peak_loc=60,
        eval_batch_size=len(self.expected_mol_dicts))

    for i in range(len(self.expected_mol_dicts)):
      self.expected_mol_dicts[i][
          fmap_constants.MOLECULE_WEIGHT] = mol_weights[i]
      self.expected_mol_dicts[i][
          fmap_constants.ATOM_WEIGHTS] = np.pad(
              np.array(atom_weights_list[i]),
              (0, self.hparams.max_atoms - len(atom_weights_list[i])),
              'constant')
      self.expected_mol_dicts[i][fmap_constants.ATOM_IDS] = np.pad(
          np.array(atom_ids_list[i]),
          (0, self.hparams.max_atoms - len(atom_ids_list[i])), 'constant')
      self.expected_mol_dicts[i][fmap_constants.ADJACENCY_MATRIX] = (
          adjacency_matrix_list[i])
      self.expected_mol_dicts[i][fmap_constants.DENSE_MASS_SPEC] = (
          feature_utils.make_dense_mass_spectra(
              mass_spec_peak_locs[i], mass_spec_peak_intensities[i],
              self.hparams.max_mass_spec_peak_loc))

  def tearDown(self):
    tf.io.gfile.rmtree(self.temp_dir)
    super(ParseSdfUtilsTest, self).tearDown()

  def encode(self, value):
    """Wrapper function for encoding strings in python 3."""
    return test_utils.encode(value, six.PY3)

  def test_get_sdf_to_mol(self):
    """Check the contents of the molecules parsed by rdkit.
    """
    mol_output = parse_sdf_utils.get_sdf_to_mol(
        self.test_file_long, max_atoms=self.hparams.max_atoms)
    self.assertLen(mol_output, 12)
    self.assertIsInstance(mol_output[0], Chem.rdchem.Mol)
    self.assertIsInstance(Chem.MolToSmiles(mol_output[0]), str)
    self.assertEqual(
        Chem.MolToSmiles(mol_output[0], isomericSmiles=True), '[H][H]')
    self.assertTrue(mol_output[0].HasProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS))

  def test_find_largest_number_of_atoms_and_largest_peak(self):
    """Test finding largest number of atoms and largest mass/charge ratio."""
    mol_output = parse_sdf_utils.get_sdf_to_mol(self.test_file_long)
    found_max_atoms, found_max_atom_num, found_max_peak_loc = (
        parse_sdf_utils.find_largest_number_of_atoms_atomic_number_and_ms_peak(
            mol_output))
    self.assertEqual(found_max_atoms, 28)
    self.assertEqual(found_max_atom_num, 35)
    self.assertEqual(found_max_peak_loc, 77)

  def test_filter_mol_list_by_prop(self):
    """Test filtering rdkit.Mol list by contents of tags."""
    mol_list = parse_sdf_utils.get_sdf_to_mol(self.test_file_long)
    filtered_mol_list = parse_sdf_utils.filter_mol_list_by_prop(
        mol_list, 'CONTRIBUTOR', 'Moscow', wanted=True)
    self.assertLen(filtered_mol_list, 9)

  def test_find_inchikey_duplicates(self):
    """Test finding duplicate inchi keys in list of molecules."""
    mol_list = parse_sdf_utils.get_sdf_to_mol(self.test_file_long)
    dup_dict = parse_sdf_utils.find_inchikey_duplicates(mol_list)
    self.assertLen(dup_dict, 1)

  def test_all_circular_fingerprints_to_dict(self):
    """Test construction of fingerprints."""
    # Test on tubocurarine chloride, which has a lot of bit collisions in its fp
    test_smiles = ('Oc7ccc1cc7Oc5cc6[C@H](Cc4ccc(Oc2c3[C@@H](C1)[N+](C)(C)'
                   'CCc3cc(OC)c2O)cc4)[N+](C)(C)CCc6cc5OC')
    test_mol = Chem.MolFromSmiles(test_smiles)

    def make_fp_key(fp_type, fp_len, rad):
      return ms_constants.CircularFingerprintKey(fp_type, fp_len, rad)

    expected_fp_sums = {
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 1024, 2):
            59.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    1024, 2):
            130.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 1024, 4):
            117.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    1024, 4):
            194.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 1024, 6):
            159.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    1024, 6):
            238.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 2048, 2):
            60.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    2048, 2):
            130.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 2048, 4):
            120.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    2048, 4):
            194.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 2048, 6):
            164.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    2048, 6):
            238.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 4096, 2):
            60.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    4096, 2):
            130.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 4096, 4):
            121.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    4096, 4):
            194.,
        make_fp_key(fmap_constants.CIRCULAR_FP_BASENAME, 4096, 6):
            165.,
        make_fp_key(fmap_constants.COUNTING_CIRCULAR_FP_BASENAME,
                    4096, 6):
            238.,
    }
    for fp_len in [1024, 2048, 4096]:
      for rad in [2, 4, 6]:
        for fp_type in fmap_constants.FP_TYPE_LIST:
          fp_key = ms_constants.CircularFingerprintKey(fp_type, fp_len, rad)
          fp = feature_utils.make_circular_fingerprint(test_mol, fp_key)
          self.assertEqual(sum(fp), expected_fp_sums[fp_key])

  def test_make_mol_dict(self):
    """Test generation of molecule dictionaries."""
    mols = parse_sdf_utils.get_sdf_to_mol(self.test_file_short)
    mol_dicts = [
        parse_sdf_utils.make_mol_dict(mol, self.hparams.max_atoms,
                                      self.hparams.max_mass_spec_peak_loc)
        for mol in mols
    ]
    for i in range(len(self.expected_mol_dicts)):
      mol_dict_key_names = [
          fmap_constants.NAME, fmap_constants.INCHIKEY,
          fmap_constants.SMILES, fmap_constants.MOLECULAR_FORMULA
      ]
      for kwarg in mol_dict_key_names:
        self.assertEqual(self.expected_mol_dicts[i][kwarg], mol_dicts[i][kwarg])
      self.assertAlmostEqual(
          self.expected_mol_dicts[i][fmap_constants.MOLECULE_WEIGHT],
          mol_dicts[i][fmap_constants.MOLECULE_WEIGHT])
      self.assertSequenceAlmostEqual(
          self.expected_mol_dicts[i][fmap_constants.ATOM_WEIGHTS],
          mol_dicts[i][fmap_constants.ATOM_WEIGHTS])
      self.assertSequenceAlmostEqual(
          self.expected_mol_dicts[i][fmap_constants.ADJACENCY_MATRIX],
          mol_dicts[i][fmap_constants.ADJACENCY_MATRIX])
      self.assertSequenceAlmostEqual(
          self.expected_mol_dicts[i][fmap_constants.DENSE_MASS_SPEC],
          mol_dicts[i][fmap_constants.DENSE_MASS_SPEC])

  def _validate_info_file(self, mol_list, fpath):
    with open(fpath + '.info') as f:
      lines = f.readlines()
      self.assertLen(lines, 1)
      self.assertLen(lines[0], len(mol_list))

  def test_dict_tfexample(self):
    """Check if the contents of tf.Records is the same as input molecule info.

       Writes tf.example as tf.record to disk, then reads from disk.
    """
    mol_list = parse_sdf_utils.get_sdf_to_mol(self.test_file_short)

    fd, fpath = tempfile.mkstemp(dir=self.temp_dir)
    os.close(fd)

    parse_sdf_utils.write_dicts_to_example(mol_list, fpath,
                                           self.hparams.max_atoms,
                                           self.hparams.max_mass_spec_peak_loc)
    parse_sdf_utils.write_info_file(mol_list, fpath)
    self._validate_info_file(mol_list, fpath)

    dataset = parse_sdf_utils.get_dataset_from_record(
        [fpath], self.hparams, mode=tf.estimator.ModeKeys.EVAL)

    feature_names = [
        fmap_constants.ATOM_WEIGHTS,
        fmap_constants.MOLECULE_WEIGHT,
        fmap_constants.DENSE_MASS_SPEC,
        fmap_constants.INCHIKEY, fmap_constants.NAME,
        fmap_constants.MOLECULAR_FORMULA,
        fmap_constants.ADJACENCY_MATRIX,
        fmap_constants.ATOM_IDS, fmap_constants.SMILES
    ]
    label_names = [fmap_constants.INCHIKEY]

    features, _ = parse_sdf_utils.make_features_and_labels(
        dataset, feature_names, label_names, mode=tf.estimator.ModeKeys.EVAL)

    with tf.Session() as sess:
      feature_values = sess.run(features)

      # Check that the dataset was consumed
      try:
        sess.run(features)
        raise ValueError('Dataset parsing using batch size of length of the'
                         'dataset resulted in more than one batch.')
      except tf.errors.OutOfRangeError:  # expected behavior
        pass

    for i in range(len(self.expected_mol_dicts)):
      self.assertAlmostEqual(
          feature_values[fmap_constants.MOLECULE_WEIGHT][i],
          self.expected_mol_dicts[i][fmap_constants.MOLECULE_WEIGHT])
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.ADJACENCY_MATRIX][i]
          .flatten(),
          self.expected_mol_dicts[i][fmap_constants.ADJACENCY_MATRIX],
          delta=0.0001)
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.DENSE_MASS_SPEC][i],
          self.expected_mol_dicts[i][fmap_constants.DENSE_MASS_SPEC],
          delta=0.0001)
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.ATOM_WEIGHTS][i],
          self.expected_mol_dicts[i][fmap_constants.ATOM_WEIGHTS],
          delta=0.0001)
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.ATOM_IDS][i],
          self.expected_mol_dicts[i][fmap_constants.ATOM_IDS],
          delta=0.0001)
      self.assertEqual(
          feature_values[fmap_constants.NAME][i],
          self.encode(self.expected_mol_dicts[i][fmap_constants.NAME]))
      self.assertEqual(
          feature_values[fmap_constants.INCHIKEY][i],
          self.encode(
              self.expected_mol_dicts[i][fmap_constants.INCHIKEY]))
      self.assertEqual(
          feature_values[fmap_constants.MOLECULAR_FORMULA][i],
          self.encode(
              self.expected_mol_dicts[i][fmap_constants.MOLECULAR_FORMULA]))
      self.assertAllEqual(feature_values[fmap_constants.SMILES][i],
                          self.expected_mol_dicts[i]['parsed_smiles'])
      self.assertAllEqual(
          feature_values[fmap_constants.SMILES_TOKEN_LIST_LENGTH][i],
          self.expected_mol_dicts[i][fmap_constants.SMILES_TOKEN_LIST_LENGTH])

  def test_save_true_spectra_array(self):
    """Checks contents of true spectra array written by write_dicts_to_example.
    """
    mol_list = parse_sdf_utils.get_sdf_to_mol(self.test_file_short)

    fpath = self.temp_dir

    records_path_name = os.path.join(fpath, 'test_record.gz')
    test_array_filename = 'true_spectra_array.npy'
    array_path_name = os.path.join(fpath, test_array_filename)

    parse_sdf_utils.write_dicts_to_example(
        mol_list,
        records_path_name,
        self.hparams.max_atoms,
        self.hparams.max_mass_spec_peak_loc,
        true_library_array_path_name=array_path_name)
    parse_sdf_utils.write_info_file(mol_list, records_path_name)

    parse_sdf_utils.validate_spectra_array_contents(
        records_path_name, self.hparams, array_path_name)

  def test_record_contents(self):
    """Test the contents of the stored record file to ensure features match."""
    mol_list = parse_sdf_utils.get_sdf_to_mol(self.test_file_long)

    mol_dicts = [parse_sdf_utils.make_mol_dict(mol) for mol in mol_list]
    parsed_smiles_tokens = [
        feature_utils.tokenize_smiles(
            np.array([mol_dict[fmap_constants.SMILES]]))
        for mol_dict in mol_dicts
    ]

    token_lengths = [
        np.shape(token_arr)[0] for token_arr in parsed_smiles_tokens
    ]
    parsed_smiles_tokens = [
        np.pad(token_arr,
               (0, ms_constants.MAX_TOKEN_LIST_LENGTH - token_length),
               'constant')
        for token_arr, token_length in zip(parsed_smiles_tokens, token_lengths)
    ]

    hparams_main = tf.contrib.training.HParams(
        max_atoms=ms_constants.MAX_ATOMS,
        max_mass_spec_peak_loc=ms_constants.MAX_PEAK_LOC,
        eval_batch_size=len(mol_list),
        intensity_power=1.0)

    dataset = parse_sdf_utils.get_dataset_from_record(
        [os.path.join(self.test_data_directory, 'test_14_record.gz')],
        hparams_main,
        mode=tf.estimator.ModeKeys.EVAL)

    feature_names = [
        fmap_constants.ATOM_WEIGHTS,
        fmap_constants.MOLECULE_WEIGHT,
        fmap_constants.DENSE_MASS_SPEC,
        fmap_constants.INCHIKEY, fmap_constants.NAME,
        fmap_constants.MOLECULAR_FORMULA,
        fmap_constants.ADJACENCY_MATRIX,
        fmap_constants.ATOM_IDS, fmap_constants.SMILES
    ]
    for fp_len in ms_constants.NUM_CIRCULAR_FP_BITS_LIST:
      for rad in ms_constants.CIRCULAR_FP_RADII_LIST:
        for fp_type in fmap_constants.FP_TYPE_LIST:
          feature_names.append(
              str(ms_constants.CircularFingerprintKey(fp_type, fp_len, rad)))
    label_names = [fmap_constants.INCHIKEY]

    features, _ = parse_sdf_utils.make_features_and_labels(
        dataset, feature_names, label_names, mode=tf.estimator.ModeKeys.EVAL)

    with tf.Session() as sess:
      feature_values = sess.run(features)

      # Check that the dataset was consumed
      try:
        sess.run(features)
        raise ValueError('Dataset parsing using batch size of length of the'
                         ' dataset resulted in more than one batch.')
      except tf.errors.OutOfRangeError:  # expected behavior
        pass

    for i in range(len(mol_list)):
      self.assertAlmostEqual(
          feature_values[fmap_constants.MOLECULE_WEIGHT][i],
          mol_dicts[i][fmap_constants.MOLECULE_WEIGHT])
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.ADJACENCY_MATRIX][i]
          .flatten(),
          mol_dicts[i][fmap_constants.ADJACENCY_MATRIX],
          delta=0.0001)
      self.assertEqual(feature_values[fmap_constants.NAME][i],
                       self.encode(mol_dicts[i][fmap_constants.NAME]))
      self.assertEqual(feature_values[fmap_constants.INCHIKEY][i],
                       self.encode(mol_dicts[i][fmap_constants.INCHIKEY]))
      self.assertEqual(
          feature_values[fmap_constants.MOLECULAR_FORMULA][i],
          self.encode(mol_dicts[i][fmap_constants.MOLECULAR_FORMULA]))
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.DENSE_MASS_SPEC][i],
          mol_dicts[i][fmap_constants.DENSE_MASS_SPEC],
          delta=0.0001)
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.ATOM_WEIGHTS][i],
          mol_dicts[i][fmap_constants.ATOM_WEIGHTS],
          delta=0.0001)
      self.assertSequenceAlmostEqual(
          feature_values[fmap_constants.ATOM_IDS][i],
          mol_dicts[i][fmap_constants.ATOM_IDS],
          delta=0.0001)
      self.assertAllEqual(feature_values[fmap_constants.SMILES][i],
                          parsed_smiles_tokens[i])
      self.assertAllEqual(
          feature_values[fmap_constants.SMILES_TOKEN_LIST_LENGTH][i],
          token_lengths[i])
      for fp_len in ms_constants.NUM_CIRCULAR_FP_BITS_LIST:
        for rad in ms_constants.CIRCULAR_FP_RADII_LIST:
          for fp_type in fmap_constants.FP_TYPE_LIST:
            fp_key = ms_constants.CircularFingerprintKey(fp_type, fp_len, rad)
            self.assertSequenceAlmostEqual(
                feature_values[str(fp_key)][i],
                mol_dicts[i][fp_key],
                delta=0.0001)


if __name__ == '__main__':
  tf.test.main()
