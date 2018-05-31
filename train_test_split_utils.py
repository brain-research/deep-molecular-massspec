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

"""Utilities for making train test split for mass spectra datsets.

Contains TrainValFractions namedtuple for passing 3-tuple of train, validation,
and test fractions to use of a datasets. Also contains TrainValTestInchikeys
namedtuple for 3-tuple of lists of inchikeys to put into train, validation, and
test splits.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import random

import feature_utils
import mass_spec_constants as ms_constants
import numpy as np

# Helper class for storing train, validation, and test fractions.
TrainValTestFractions = namedtuple('TrainValTestFractions',
                                   ['train', 'validation', 'test'])

# Helper class for storing train, validation, and test inchikeys after split.
TrainValTestInchikeys = namedtuple('TrainValTestInchikeys',
                                   ['train', 'validation', 'test'])


def assert_all_lists_mutally_exclusive(list_of_lists):
  """Check if any lists within a list of lists contain identical items."""
  for idx, list1 in enumerate((list_of_lists)):
    for list2 in list_of_lists[idx + 1:]:
      if any(elem in list2 for elem in list1):
        raise ValueError(
            'found matching items between two lists: \n {}\n {}'.format(
                ', '.join(list1),
                ', '.join(list2),
            ))


def make_inchikey_dict(mol_list):
  """Converts rdkit.Mol list into dict of lists of Mols keyed by inchikey."""
  inchikey_dict = {}
  for mol in mol_list:
    inchikey = mol.GetProp(ms_constants.SDF_TAG_INCHIKEY)
    if inchikey not in inchikey_dict:
      inchikey_dict[inchikey] = [mol]
    else:
      inchikey_dict[inchikey].append(mol)
  return inchikey_dict


def get_random_inchikeys(inchikey_list, train_val_test_split_fractions):
  """Splits a given inchikey list of into 3 lists for train/val/test sets."""
  random.shuffle(inchikey_list)

  train_num = int(train_val_test_split_fractions.train * len(inchikey_list))
  val_num = int(train_val_test_split_fractions.validation * len(inchikey_list))

  return TrainValTestInchikeys(inchikey_list[:train_num],
                               inchikey_list[train_num:train_num + val_num],
                               inchikey_list[train_num + val_num:])


def get_inchikeys_by_family(inchikey_list,
                            inchikey_dict,
                            train_val_test_split_fractions,
                            family_name='steroid',
                            exclude_from_train=True):
  """Creates train/val/test split based on presence of steroids.

  Filters molecules according to whether they have the substructure specified
  by family_name. All molecules passing the filter will be placed in
  validation/test datasets or into the train set according to exclude from
  train. The molecules assigned to the validation/test split according to the
  relative ratio between the validation/test fractions.

  If the validation and tests fractions are both equal to 0.0, these values
  will be over written to 0.5 and 0.5.

  Args:
    inchikey_list: List of inchikeys to partition into train/val/test sets
    inchikey_dict: dict of inchikeys, [rdkit.Mol objects].
        Must contain inchikey_list in its keys.
    train_val_test_split_fractions: a TrainValTestFractions tuple
    family_name: str, a key in feature_utils.FAMILY_DICT
    exclude_from_train: indicates whether to include/exclude steroid molecules
        from training set. If excluded from training set, test and validation
        sets will be comprised only of these molecules.
  Returns:
    TrainValTestInchikeys object
  """
  _, val_fraction, test_fraction = train_val_test_split_fractions
  if val_fraction == 0.0 and test_fraction == 0.0:
    val_fraction = 0.5
    test_fraction = 0.5

  substructure_filter_fn = feature_utils.make_filter_by_substructure(
      family_name)
  family_inchikeys = []
  nonfamily_inchikeys = []

  for ikey in inchikey_list:
    if substructure_filter_fn(inchikey_dict[ikey][0]):
      family_inchikeys.append(ikey)
    else:
      nonfamily_inchikeys.append(ikey)

  if exclude_from_train:
    val_test_inchikeys, train_inchikeys = (family_inchikeys,
                                           nonfamily_inchikeys)
  else:
    train_inchikeys, val_test_inchikeys = (family_inchikeys,
                                           nonfamily_inchikeys)

  random.shuffle(val_test_inchikeys)
  val_num = int(
      val_fraction / (val_fraction + test_fraction) * len(val_test_inchikeys))
  return TrainValTestInchikeys(train_inchikeys, val_test_inchikeys[:val_num],
                               val_test_inchikeys[val_num:])


def make_train_val_test_split_inchikey_lists(train_inchikey_list,
                                             train_inchikey_dict,
                                             train_val_test_split_fractions,
                                             holdout_inchikey_list=None,
                                             splitting_type='random'):
  """Given inchikey lists, returns lists to use for train/val/test sets.

  If holdout_inchikey_list is given, the inchikeys in this list will be excluded
  from the returned train/validation/test lists.

  Args:
    train_inchikey_list : List of inchikeys to use for train/val/test sets
    train_inchikey_dict : Main dict keyed by inchikeys, values are lists of
        rdkit.Mol. Note that train_inchikey_dict.keys() != train_inchikey_list
        train_inchikey_dict will have many more keys than are in the list.
    train_val_test_split_fractions : a TrainValTestFractions tuple
    holdout_inchikey_list : List of inchikeys to exclude from train/val/test
                            sets.
    splitting_type : method of splitting molecules into train/val/test sets.
  Returns:
    A TrainValTestInchikeys namedtuple
  Raises:
    ValueError : if not train_val_test_split_sizes XOR
                        train_val_test_split_fractions
                 or if specify a splitting_type that isn't implemented yet.
  """
  if not np.isclose([sum(train_val_test_split_fractions)], [1.0]):
    raise ValueError('Must specify train_val_test_split that sums to 1.0')

  if holdout_inchikey_list:
    # filter out those inchikeys that are in the holdout set.
    train_inchikey_list = [
        ikey for ikey in train_inchikey_list
        if ikey not in holdout_inchikey_list
    ]

  if splitting_type == 'random':
    return get_random_inchikeys(train_inchikey_list,
                                train_val_test_split_fractions)
  else:
    # Assume that splitting_type is the name of a structure family.
    # get_inchikeys_by_family will throw an error if this is not supported.
    return get_inchikeys_by_family(
        train_inchikey_list,
        train_inchikey_dict,
        train_val_test_split_fractions,
        family_name=splitting_type,
        exclude_from_train=True)


def make_mol_list_from_inchikey_dict(inchikey_dict, inchikey_list):
  """Return a list of rdkit.Mols given a list of inchikeys.

  Args:
     inchikey_dict : a dict of lists of rdkit.Mol objects keyed by inchikey
     inchikey_list : List of inchikeys of molecules we want in a list.
  Returns:
     A list of rdkit.Mols corresponding to inchikeys in inchikey_list.
  """
  mol_list = []
  for inchikey in inchikey_list:
    mol_list.extend(inchikey_dict[inchikey])

  return mol_list
