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

"""Module containing all commonly used variables in this repo."""

from collections import namedtuple


class CircularFingerprintKey(
    namedtuple('CircularFingerprintKey', ['fp_type', 'fp_len', 'radius'])):
  """Helper function for labeling fingerprint keys."""

  def __str__(self):
    return self.fp_type + '_' + str(self.fp_len) + '_' + str(self.radius)


# Constants for SDF tags found in NIST sdf files:
SDF_TAG_MASS_SPEC_PEAKS = 'MASS SPECTRAL PEAKS'
SDF_TAG_INCHIKEY = 'INCHIKEY'
SDF_TAG_NAME = 'NAME'
SDF_TAG_MOLECULE_MASS = 'EXACT MASS'
SDF_TAG_MASS_SPEC_PEAKS = 'MASS SPECTRAL PEAKS'

# Constants for fields in TFRecords
MAX_MZ_WEIGHT_RATIO = 3.0
MAX_PEAK_LOC = 1000
MAX_ATOMS = 100
MAX_ATOM_ID = 100
MAX_TOKEN_LIST_LENGTH = 230

CIRCULAR_FP_RADII_LIST = [2, 4, 6]
NUM_CIRCULAR_FP_BITS_LIST = [1024, 2048, 4096]
ADD_HS_TO_MOLECULES = False

TWO_LETTER_TOKEN_NAMES = [
    'Al', 'Ce', 'Co', 'Ge', 'Gd', 'Cs', 'Th', 'Cd', 'As', 'Na', 'Nb', 'Li',
    'Ni', 'Se', 'Sc', 'Sb', 'Sn', 'Hf', 'Hg', 'Si', 'Be', 'Cl', 'Rb', 'Fe',
    'Bi', 'Br', 'Ag', 'Ru', 'Zn', 'Te', 'Mo', 'Pt', 'Mn', 'Os', 'Tl', 'In',
    'Cu', 'Mg', 'Ti', 'Pb', 'Re', 'Pd', 'Ir', 'Rh', 'Zr', 'Cr', '@@', 'se',
    'si', 'te'
]

METAL_ATOM_SYMBOLS = [
    'As', 'Cr', 'Cs', 'Cu', 'Be', 'Ag', 'Co', 'Al', 'Cd', 'Ce', 'Si', 'Sn',
    'Os', 'Sb', 'Sc', 'In', 'Se', 'Ni', 'Th', 'Hg', 'Hf', 'Li', 'Nb', 'U', 'Y',
    'V', 'W', 'Tl', 'Na', 'Fe', 'K', 'Zr', 'B', 'Pb', 'Pd', 'Rh', 'Re', 'Gd',
    'Ge', 'Ir', 'Rb', 'Ti', 'Pt', 'Mn', 'Mg', 'Ru', 'Bi', 'Zn', 'Te', 'Mo'
]

SMILES_TOKEN_NAMES = [
    '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '=', '@', '@@', 'Ag', 'Al', 'As', 'B', 'Be', 'Bi', 'Br', 'C',
    'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'F', 'Fe', 'Gd', 'Ge', 'H', 'Hf',
    'Hg', 'I', 'In', 'Ir', 'K', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Ni',
    'O', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc',
    'Se', 'Si', 'Sn', 'Te', 'Th', 'Ti', 'Tl', 'U', 'V', 'W', 'Y', 'Zn', 'Zr',
    '[', '\\', ']', 'c', 'n', 'o', 'p', 's'
]

SMILES_TOKEN_NAME_TO_INDEX = {
    name: idx for idx, name in enumerate(SMILES_TOKEN_NAMES)
}

# Add 3 elements which also have lowercase representations in SMILES string.
# We want these to have the same index as the upper-lower case version.
SMILES_TOKEN_NAME_TO_INDEX['se'] = SMILES_TOKEN_NAME_TO_INDEX['Se']
SMILES_TOKEN_NAME_TO_INDEX['si'] = SMILES_TOKEN_NAME_TO_INDEX['Si']
SMILES_TOKEN_NAME_TO_INDEX['te'] = SMILES_TOKEN_NAME_TO_INDEX['Te']

# Bond order master list:
BOND_ORDER_TO_INTS_DICT = {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}

TRUE_SPECTRA_SCALING_FACTOR = 0.1
