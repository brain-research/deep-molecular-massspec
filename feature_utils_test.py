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

"""Tests for feature_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import feature_utils
import mass_spec_constants as ms_constants
import numpy as np
from rdkit import Chem
import tensorflow as tf


class FeatureUtilsTest(tf.test.TestCase):

  def setUp(self):
    test_mol_smis = [
        'CCCC', 'CC(=CC(=O)C)O.CC(=CC(=O)C)O.[Cu]', 'CCCCl',
        ('C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1'
         '(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C')
    ]
    self.test_mols = [
        Chem.MolFromSmiles(mol_str) for mol_str in test_mol_smis
    ]

  def _validate_smiles_string_tokenization(self, smiles_string,
                                           expected_token_list):
    token_list = feature_utils.tokenize_smiles(np.array([smiles_string]))
    self.assertAllEqual(token_list, expected_token_list)

  def test_tokenize_smiles_string(self):
    self._validate_smiles_string_tokenization('CCCC', [28, 28, 28, 28])
    self._validate_smiles_string_tokenization('ClCCCC', [31, 28, 28, 28, 28])
    self._validate_smiles_string_tokenization('CCClCC', [28, 28, 31, 28, 28])
    self._validate_smiles_string_tokenization('CCCCCl', [28, 28, 28, 28, 31])
    self._validate_smiles_string_tokenization('ClC(CC)CCl',
                                              [31, 28, 2, 28, 28, 3, 28, 31])
    self._validate_smiles_string_tokenization(
        'ClC(CCCl)CCl', [31, 28, 2, 28, 28, 31, 3, 28, 31])
    self._validate_smiles_string_tokenization('BrCCCCCl',
                                              [27, 28, 28, 28, 28, 31])
    self._validate_smiles_string_tokenization('ClCCCCBr',
                                              [31, 28, 28, 28, 28, 27])
    self._validate_smiles_string_tokenization('[Te][te]',
                                              [81, 71, 83, 81, 71, 83])

  def test_check_mol_only_has_atoms(self):
    result = [
        feature_utils.check_mol_only_has_atoms(mol, ['C'])
        for mol in self.test_mols
    ]
    self.assertAllEqual(result, [True, False, False, False])

  def test_check_mol_does_not_have_atoms(self):
    result = [
        feature_utils.check_mol_does_not_have_atoms(
            mol, ms_constants.METAL_ATOM_SYMBOLS) for mol in self.test_mols
    ]
    self.assertAllEqual(result, [True, False, True, True])

  def test_make_filter_by_substructure(self):
    filter_fn = feature_utils.make_filter_by_substructure('steroid')
    result = [filter_fn(mol) for mol in self.test_mols]
    self.assertAllEqual(result, [False, False, False, True])

  def test_convert_spectrum_array_to_string(self):
    spectra_array = np.zeros((2, 1000))
    spectra_array[0, 3] = 100
    spectra_array[1, 39] = 100
    spectra_array[1, 21] = 60

    expected_spectra_strings = ['3 100', '21 60\n39 100']
    result_spectra_strings = []
    for idx in range(np.shape(spectra_array)[0]):
      result_spectra_strings.append(
          feature_utils.convert_spectrum_array_to_string(spectra_array[idx, :]))

    self.assertAllEqual(expected_spectra_strings, result_spectra_strings)


if __name__ == '__main__':
  tf.test.main()
