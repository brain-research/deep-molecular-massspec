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

"""Module containing the feature names used in the TFRecords of this repo."""

# Overall Tasks
LIBRARY_MATCHING = 'library_matching'
SPECTRUM_PREDICTION = 'spectrum_prediction'

# Feature names
ATOM_WEIGHTS = 'atom_weights'
MOLECULE_WEIGHT = 'molecule_weight'
CIRCULAR_FP_BASENAME = 'circular_fp'
COUNTING_CIRCULAR_FP_BASENAME = 'counting_circular_fp'
INDEX_TO_GROUND_TRUTH_ARRAY = 'index_in_library'

FP_TYPE_LIST = [
    CIRCULAR_FP_BASENAME, COUNTING_CIRCULAR_FP_BASENAME
]

DENSE_MASS_SPEC = 'dense_mass_spec'
INCHIKEY = 'inchikey'
NAME = 'name'
SMILES = 'smiles'
MOLECULAR_FORMULA = 'molecular_formula'
ADJACENCY_MATRIX = 'adjacency_matrix'
ATOM_IDS = 'atom_id'
SMILES_TOKEN_LIST_LENGTH = 'smiles_sequence_len'
