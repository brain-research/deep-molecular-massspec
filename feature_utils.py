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

"""Functions for getting molecule features from NIST sdf data.

This module includes functions to help with parsing sdf files and generating
features such as atom weight lists and adjacency matrices. Also contains a
function to parse mass spectra peaks from their string format in the NIST sdf
files.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import feature_map_constants as fmap_constants
import mass_spec_constants as ms_constants
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

FILTER_DICT = {
    'steroid':
        Chem.MolFromSmarts(
            '[#6]1~[#6]2~[#6](~[#6]~[#6]~[#6]~1)'
            '~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~1~[#6]~[#6]~[#6]~2'
        ),
    'diazo':
        Chem.MolFromSmarts(
            '[#7]~[#7]'
        ),
}


def get_smiles_string(mol):
  """Make canonicalized smiles from rdkit.Mol."""
  return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def get_molecular_formula(mol):
  """Makes string of molecular formula from rdkit.Mol."""
  return AllChem.CalcMolFormula(mol)


def parse_peaks(pk_str):
  r"""Helper function for converting peak string into vector form.

  Args:
    pk_str : String from NIST MS data of format
      "peak1_loc peak1_int\npeak2_loc peak2_int"

  Returns:
    A tuple containing two arrays: (peak_locs, peak_ints)
    peak_locs : int list of the location of peaks
    peak_intensities : float list of the intensity of the peaks
  """
  all_peaks = pk_str.split('\n')

  peak_locs = []
  peak_intensities = []

  for peak in all_peaks:
    loc, intensity = peak.split()
    peak_locs.append(int(loc))
    peak_intensities.append(float(intensity))

  return peak_locs, peak_intensities


def convert_spectrum_array_to_string(spectrum_array):
  """Write a spectrum array to string.

  Args:
    spectrum_array : np.array of shape (1000)

  Returns:
    string representing the peaks of the spectra.
  """
  mz_peak_locations = np.nonzero(spectrum_array)[0].tolist()
  mass_peak_strings = [
      '%d %d' % (p, spectrum_array[p]) for p in mz_peak_locations
  ]
  return '\n'.join(mass_peak_strings)


def get_largest_mass_spec_peak_loc(mol):
  """Returns largest ms peak location from an rdkit.Mol object."""
  return parse_peaks(mol.GetProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS))[0][-1]


def make_dense_mass_spectra(peak_locs, peak_intensities, max_peak_loc):
  """Make a dense np.array of the mass spectra.

  Args:
    peak_locs : int list of the location of peaks
    peak_intensities : float list of the intensity of the peaks
    max_peak_loc : maximum number of peaks bins

  Returns:
    np.array of the mass spectra data as a dense vector.
  """
  dense_spectrum = np.zeros(max_peak_loc)
  dense_spectrum[peak_locs] = peak_intensities

  return dense_spectrum


def get_padded_atom_weights(mol, max_atoms):
  """Make a padded list of atoms of length max_atoms given rdkit.Mol object.

  Note: Returns atoms in the same order as the input rdkit.Mol.
        If you want the atoms in canonical order, you should canonicalize
        the molecule first.

  Args:
    mol : a rdkit.Mol object
    max_atoms : maximum number of atoms to consider
  Returns:
    np array of atoms by atomic mass of shape (max_atoms)
  Raises:
    ValueError : If rdkit.Mol object had more atoms than max_atoms.
  """

  if max_atoms < len(mol.GetAtoms()):
    raise ValueError(
        'molecule contains {} atoms, more than max_atoms {}'.format(
            len(mol.GetAtoms()), max_atoms))

  atom_list = np.array([at.GetMass() for at in mol.GetAtoms()])
  atom_list = np.pad(atom_list, ((0, max_atoms - len(atom_list))), 'constant')
  return atom_list


def get_padded_atom_ids(mol, max_atoms):
  """Make a padded list of atoms of length max_atoms given rdkit.Mol object.

  Args:
    mol : a rdkit.Mol object
    max_atoms : maximum number of atoms to consider
  Returns:
    np array of atoms by atomic number of shape (max_atoms)
    Note: function returns atoms in the same order as the input rdkit.Mol.
          If you want the atoms in canonical order, you should canonicalize
          the molecule first.
  Raises:
    ValueError : rdkit.Mol object is too big, had more atoms than max_atoms.
  """
  if max_atoms < len(mol.GetAtoms()):
    raise ValueError(
        'molecule contains {} atoms, more than max_atoms {}'.format(
            len(mol.GetAtoms()), max_atoms))
  atom_list = np.array([at.GetAtomicNum() for at in mol.GetAtoms()])
  atom_list = atom_list.astype('int32')
  atom_list = np.pad(atom_list, ((0, max_atoms - len(atom_list))), 'constant')

  return atom_list


def get_padded_adjacency_matrix(mol, max_atoms, add_hs_to_molecule=False):
  """Make a matrix with shape (max_atoms, max_atoms) given rdkit.Mol object.

  Args:
    mol: a rdkit.Mol object
    max_atoms : maximum number of atoms to consider
    add_hs_to_molecule : whether or not to add hydrogens to the molecule.
  Returns:
    np.array of floats of a flattened adjacency matrix of length
    (max_atoms * max_atoms)
    The values will be the index of the bond order in the alphabet
  Raises:
    ValueError : rdkit.Mol object is too big, had more atoms than max_atoms.
  """
  # Add hydrogens to atoms:
  if add_hs_to_molecule:
    mol = Chem.rdmolops.AddHs(mol)

  num_atoms_in_mol = len(mol.GetAtoms())
  if max_atoms < num_atoms_in_mol:
    raise ValueError(
        'molecule contains {} atoms, more than max_atoms {}'.format(
            len(mol.GetAtoms()), max_atoms))

  adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)

  for i in range(np.shape(adj_matrix)[0]):
    for j in range(np.shape(adj_matrix)[1]):
      if adj_matrix[i, j] != 0:
        adj_matrix[i, j] = ms_constants.BOND_ORDER_TO_INTS_DICT[adj_matrix[i,
                                                                           j]]

  padded_adjacency_matrix = np.zeros((max_atoms, max_atoms))

  padded_adjacency_matrix[:num_atoms_in_mol, :num_atoms_in_mol] = adj_matrix
  padded_adjacency_matrix = padded_adjacency_matrix.astype('int32')

  return np.reshape(padded_adjacency_matrix, (max_atoms * max_atoms))


def make_circular_fingerprint(mol, circular_fp_key):
  """Returns circular fingerprint for a mol given its circular_fp_key.

  Args:
    mol : rdkit.Mol
    circular_fp_key : A ms_constants.CircularFingerprintKey object
  Returns:
    np.array of len circular_fp_key.fp_len
  """
  # A dictionary to record rdkit functions to base names
  fp_methods_dict = {
      fmap_constants.CIRCULAR_FP_BASENAME:
          AllChem.GetMorganFingerprintAsBitVect,
      fmap_constants.COUNTING_CIRCULAR_FP_BASENAME:
          AllChem.GetHashedMorganFingerprint
  }

  fp = fp_methods_dict[circular_fp_key.fp_type](
      mol, circular_fp_key.radius, nBits=circular_fp_key.fp_len)
  fp_arr = np.zeros(1)
  DataStructs.ConvertToNumpyArray(fp, fp_arr)
  return fp_arr


def all_circular_fingerprints_to_dict(mol):
  """Creates all circular fingerprints from list of lengths and radii.

  Based on lists of fingerprint lengths and fingerprint radii inside
  mass_spec_constants.

  Args:
    mol : rdkit.Mol
  Returns:
    a dict. The keys are CircularFingerprintKey instances and the values are
    the corresponding fingerprints
  """
  fp_dict = {}
  for fp_len in ms_constants.NUM_CIRCULAR_FP_BITS_LIST:
    for rad in ms_constants.CIRCULAR_FP_RADII_LIST:
      for fp_type in fmap_constants.FP_TYPE_LIST:
        circular_fp_key = ms_constants.CircularFingerprintKey(
            fp_type, fp_len, rad)
        fp_dict[circular_fp_key] = make_circular_fingerprint(
            mol, circular_fp_key)
  return fp_dict


def check_mol_has_non_empty_smiles(mol):
  """Checks if smiles string of rdkit.Mol is an empty string."""
  return bool(get_smiles_string(mol))


def check_mol_has_non_empty_mass_spec_peak_tag(mol):
  """Checks if mass spec sdf tag is in properties of rdkit.Mol."""
  return ms_constants.SDF_TAG_MASS_SPEC_PEAKS in mol.GetPropNames()


def check_mol_only_has_atoms(mol, accept_atom_list):
  """Checks if rdkit.Mol only contains atoms from accept_atom_list."""
  atom_symbol_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
  return all(atom in accept_atom_list for atom in atom_symbol_list)


def check_mol_does_not_have_atoms(mol, exclude_atom_list):
  """Checks if rdkit.Mol contains any molecule from exclude_atom_list."""
  atom_symbol_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
  return all(atom not in atom_symbol_list for atom in exclude_atom_list)


def check_mol_has_substructure(mol, substructure_mol):
  """Checks if rdkit.Mol has substructure.

  Args:
    mol : rdkit.Mol, representing query
    substructure_mol: rdkit.Mol, representing substructure family
  Returns:
    Boolean, True if substructure found in molecule.
  """
  return mol.HasSubstructMatch(substructure_mol)


def make_filter_by_substructure(family_name):
  """Returns a filter function according to the family_name."""
  if family_name not in FILTER_DICT.keys():
    raise ValueError('%s is not supported for family splitting' % family_name)
  return lambda mol: check_mol_has_substructure(mol, FILTER_DICT[family_name])


def tokenize_smiles(smiles_string_arr):
  """Creates a list of tokens from a smiles string.

  Two letter atom characters are considered to be a single token.
  All two letter tokens observed in this dataset are recorded in
  ms_constants.TWO_LETTER_TOKEN_NAMES.

  Args:
    smiles_string_arr: np.array of dtype str and shape (1, )
  Returns:
    A np.array of ints corresponding with the tokens
  """

  smiles_str = smiles_string_arr[0]
  if isinstance(smiles_str, bytes):
    smiles_str = smiles_str.decode('utf-8')

  token_list = []
  ptr = 0

  while ptr < len(smiles_str):
    if smiles_str[ptr:ptr + 2] in ms_constants.TWO_LETTER_TOKEN_NAMES:
      token_list.append(
          ms_constants.SMILES_TOKEN_NAME_TO_INDEX[smiles_str[ptr:ptr + 2]])
      ptr += 2
    else:
      token_list.append(
          ms_constants.SMILES_TOKEN_NAME_TO_INDEX[smiles_str[ptr]])
      ptr += 1

  return np.array(token_list, dtype=np.int64)
