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

"""Library of functions to parse SDF files into tf.Examples.

This module contains functions to for parsing SDF files into rdkit.Mol objects,
rdkit.Mol objects to dictionaries, and dictionaries to tf.Examples, which is
written to a tf.record.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections


from absl import logging
import feature_map_constants as fmap_constants
import feature_utils
import mass_spec_constants as ms_constants
import numpy as np
from rdkit import Chem
import six
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 200000


def find_inchikey_duplicates(mol_list):
  """Analyze a list of rdkit.Mol and identify InChIKey with more than molecule.

  These duplicates usually denote stereoisomers.

  Args:
    mol_list : List rdkit.Mol objects to analyze

  Returns:
    dup_dict : dictionary of duplicates, dup_dict[inchi_key] = freq of key
  """
  inchi_key_list = [
      mol.GetProp(ms_constants.SDF_TAG_INCHIKEY) for mol in mol_list
  ]
  if len(inchi_key_list) == len(set(inchi_key_list)):
    return {}

  count = collections.Counter(inchi_key_list)
  dup_dict = dict((kwarg, count[kwarg]) for kwarg in count if count[kwarg] > 1)
  logging.info('Found %s duplicates', str(len(dup_dict)))
  return dup_dict


def get_sdf_to_mol(
    sdf_fname,
    fail_sdf_fname=None,
    max_atoms=ms_constants.MAX_ATOMS,
    max_mass_peak_loc=ms_constants.MAX_PEAK_LOC,
    filter_max_mass_charge_peak_weight_cutoff=ms_constants.MAX_MZ_WEIGHT_RATIO):
  """Parse sdf file into rdkit.Mol objects. Record failing molblocks.

  Note: rdkit's Chem.SDMolSupplier only accepts filenames as inputs. As such
  this code only supports local filesystem name environments.

  Args:
    sdf_fname : Path to input sdf file
    fail_sdf_fname : Path to write failed molblocks, if set.
    max_atoms : Maximum number of atoms to allow in a molecule to be recorded.
                Set to None to return all molecules parsed by SDMolSupplier.
    max_mass_peak_loc : Largest peak position to allow in returned list.
    filter_max_mass_charge_peak_weight_cutoff :
        a float. Filters out molecules whose largest mass-to-charge peak
        is greater than the molecular weight by this cutoff value.
        To turn off this behavior, set this value to None.

  Returns:
    mols : list of rdkit.Mol objects parsed from sdf

  Raises:
    ValueError : if first mol block does not contain 'M  END' in the mol block
  """
  suppl = Chem.SDMolSupplier(sdf_fname)

  # Adding a quick test to see if necessary 'M  END' line is present in first
  # molecule block.
  if 'M  END' not in suppl.GetItemText(0):
    raise ValueError('First molecule in sdf does not contain M  END text.')

  mols = []
  num_failed = 0
  fail_sdf_blocks = ''

  for idx, mol in enumerate(suppl):
    if mol is not None and (mol.GetNumAtoms() <= max_atoms or
                            max_atoms is None):
      mols.append(mol)
    else:
      num_failed += 1
      if fail_sdf_fname is not None:
        fail_sdf_blocks += suppl.GetItemText(idx)

  # Save failed sdf blocks to disk if pathname specified.
  if fail_sdf_fname is not None and fail_sdf_blocks is not None:
    with tf.gfile.Open(fail_sdf_fname, 'w') as f:
      f.write(fail_sdf_blocks)

  if num_failed:
    logging.warn('Number of failed molblocks : %s', num_failed)

  def _mol_passes_filters(mol):
    """A helper function for testing mols on all filtering conditions."""
    if not feature_utils.check_mol_has_non_empty_mass_spec_peak_tag(mol):
      return False

    filter_list = []
    filter_list.append(feature_utils.check_mol_has_non_empty_smiles(mol))

    if filter_max_mass_charge_peak_weight_cutoff is not None:
      # Parse largest spectral peak location and mass from molecule properties.
      charge_weight_ratio = (
          feature_utils.get_largest_mass_spec_peak_loc(mol) / float(
              mol.GetProp(ms_constants.SDF_TAG_MOLECULE_MASS)))

      filter_list.append(
          charge_weight_ratio < filter_max_mass_charge_peak_weight_cutoff)

    filter_list.append(
        feature_utils.get_largest_mass_spec_peak_loc(mol) < max_mass_peak_loc)

    return all(filter_list)

  mols = [mol for mol in mols if _mol_passes_filters(mol)]
  return mols


def filter_mol_list_by_prop(mol_list, prop_key, prop_value, wanted=True):
  """Allows for filtering rdkit.Mol list based on property values.

  Args:
    mol_list : list of rdkit.Mol objects to filter
    prop_key : Name of tag in SDF file to filter on.
    prop_value : Value of tag in SDF file to filter on.
    wanted : if true, include values specified by prop_value
             if false, exclude values specified by prop_value
  Returns:
    a filtered mol list.
  """
  if wanted:
    return [
        mol for mol in mol_list
        if mol.HasProp(prop_key) and prop_value in mol.GetProp(prop_key)
    ]
  else:
    return [
        mol for mol in mol_list
        if not mol.HasProp(prop_key) or prop_value not in mol.GetProp(prop_key)
    ]


def find_largest_number_of_atoms_atomic_number_and_ms_peak(
    mol_list, add_hs_to_molecule=False):
  """Finds the greatest number of molecules and the largest peak in the spectra.

  Used to determine the largest number of atoms in a molecule, and the
  largest mass/charge ratio found in the spectra.

  Args:
    mol_list : list of rdkit.Mol objects
    add_hs_to_molecule : whether or not to add hydrogens to the molecule.

  Returns:
    max_atoms : Largest number of atoms found in a molecule in mol_list
    max_atom_type: Largest atomic number in all the molecules from the mol_list
    max_peak_loc : Greatest mass/charge peak found in samples in sdf_file

  Raises:
    ValueError: If first rdkit.Mol in mol_list does not have
        MASS SPECTRAL PEAKS as one of the properties.
        This should already be present as one of the tags in the original
        sdf file the rdkit.Mols were parsed from.
  """
  if not mol_list[0].HasProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS):
    raise ValueError('first molecule in list does not contain SDF tag'
                     '\'{}\''.format(ms_constants.SDF_TAG_MASS_SPEC_PEAKS))

  # Add hydrogens to atoms:
  if add_hs_to_molecule:
    mol_list = [Chem.rdmolops.AddHs(mol) for mol in mol_list]

  max_atoms = max(mol.GetNumAtoms() for mol in mol_list)
  max_atom_type = max(
      [at.GetAtomicNum() for mol in mol_list for at in mol.GetAtoms()])

  max_peak_loc = max(
      feature_utils.get_largest_mass_spec_peak_loc(mol) for mol in mol_list)

  return max_atoms, max_atom_type, max_peak_loc


def make_mol_dict(mol,
                  max_atoms=ms_constants.MAX_ATOMS,
                  max_mass_spec_peak_loc=ms_constants.MAX_PEAK_LOC,
                  add_hs_to_molecule=ms_constants.ADD_HS_TO_MOLECULES):
  """Place molecule information from rdkit.Mol.Prop in a dictionary.

  Reads information from an rdkit mol with the desired tags for reading into
  a dictionary.

  Args:
    mol : An rdkit.Mol object read from an sdf that contains sdf tags:
             fmap_constants.NAME, 'INCHIKEY', 'MASS SPECTRAL PEAKS'
          to be converted into dictionary keys. These tags should be stored as
          properties by the rdkit.Mol
    max_atoms : maximum number of allowed atoms.
    max_mass_spec_peak_loc : largest mass/charge ratio to allow in a spectra
    add_hs_to_molecule : Whether or not to include hydrogen atoms in the
        molecule

  Returns:
    mol_dict : A dictionary with following molecule info
          keys : name, inchi_key, circular_fp_1024, molecule_weight, smiles,
          atom_weights, mass_spec_dense

  Notes:
    - smiles recorded in mol_dict here have been canonicalized, and may not
      correspond with the order in the original SDF
    - atom_weights vector are length max_atoms
    - circular_fp_1024 has 1024 bits, with default radius of 2.
  """
  # For canonicalizing molecules
  smiles_canon = feature_utils.get_smiles_string(mol)
  mol_formula = feature_utils.get_molecular_formula(mol)
  mol_canon = Chem.MolFromSmiles(smiles_canon)
  if add_hs_to_molecule:
    mol_canon = Chem.AddHs(mol_canon)

  mass_spec_locs, mass_spec_intensities = feature_utils.parse_peaks(
      mol.GetProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS))
  mass_spec_dense_vec = feature_utils.make_dense_mass_spectra(
      mass_spec_locs, mass_spec_intensities, max_mass_spec_peak_loc)

  atom_wts = feature_utils.get_padded_atom_weights(mol_canon, max_atoms)
  atom_ids = feature_utils.get_padded_atom_ids(mol_canon, max_atoms)
  adjacency_matrix = feature_utils.get_padded_adjacency_matrix(
      mol_canon, max_atoms)

  mol_dict = {
      fmap_constants.NAME:
          mol.GetProp(ms_constants.SDF_TAG_NAME),
      fmap_constants.INCHIKEY:
          mol.GetProp(ms_constants.SDF_TAG_INCHIKEY),
      fmap_constants.MOLECULAR_FORMULA: mol_formula,
      fmap_constants.MOLECULE_WEIGHT:
          float(mol.GetProp(ms_constants.SDF_TAG_MOLECULE_MASS)),
      fmap_constants.SMILES:
          smiles_canon,
      fmap_constants.ATOM_WEIGHTS:
          atom_wts,
      fmap_constants.ATOM_IDS:
          atom_ids,
      fmap_constants.DENSE_MASS_SPEC:
          mass_spec_dense_vec,
      fmap_constants.ADJACENCY_MATRIX:
          adjacency_matrix,
  }

  # Making fingerprint features:
  mol_dict.update(feature_utils.all_circular_fingerprints_to_dict(mol))

  return mol_dict


def dict_to_tfexample(mol_dict):
  """Convert dictionary of molecular info to tfExample.

  Args:
    mol_dict : dictionary containing molecule info.

  Returns:
    example : tf.example containing mol_dict info.
  """
  example = tf.train.Example()
  feature_map = example.features.feature
  feature_map[fmap_constants.ATOM_WEIGHTS].float_list.value.extend(
      mol_dict[fmap_constants.ATOM_WEIGHTS])
  feature_map[fmap_constants.ATOM_IDS].int64_list.value.extend(
      mol_dict[fmap_constants.ATOM_IDS])
  feature_map[fmap_constants.ADJACENCY_MATRIX].int64_list.value.extend(
      mol_dict[fmap_constants.ADJACENCY_MATRIX])
  feature_map[fmap_constants.MOLECULE_WEIGHT].float_list.value.append(
      mol_dict[fmap_constants.MOLECULE_WEIGHT])
  feature_map[fmap_constants.DENSE_MASS_SPEC].float_list.value.extend(
      mol_dict[fmap_constants.DENSE_MASS_SPEC])
  feature_map[fmap_constants.INCHIKEY].bytes_list.value.append(
      mol_dict[fmap_constants.INCHIKEY].encode('utf-8'))
  feature_map[fmap_constants.MOLECULAR_FORMULA].bytes_list.value.append(
      mol_dict[fmap_constants.MOLECULAR_FORMULA].encode('utf-8'))
  feature_map[fmap_constants.NAME].bytes_list.value.append(
      mol_dict[fmap_constants.NAME].encode('utf-8'))
  feature_map[fmap_constants.SMILES].bytes_list.value.append(
      mol_dict[fmap_constants.SMILES].encode('utf-8'))

  if fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY in mol_dict:
    feature_map[
        fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY].int64_list.value.append(
            mol_dict[fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY])

  for fp_len in ms_constants.NUM_CIRCULAR_FP_BITS_LIST:
    for rad in ms_constants.CIRCULAR_FP_RADII_LIST:
      for fp_type in fmap_constants.FP_TYPE_LIST:
        fp_key = ms_constants.CircularFingerprintKey(fp_type, fp_len, rad)
        feature_map[str(fp_key)].float_list.value.extend(mol_dict[fp_key])

  return example


def write_dicts_to_example(mol_list,
                           record_path_name,
                           max_atoms,
                           max_mass_spec_peak_loc,
                           true_library_array_path_name=None):
  """Helper function for writing tf.record from all examples.

  Uses dict_to_tfexample to write the actual tf.example

  Args:
    mol_list : list of rdkit.Mol objects
    record_path_name : file name for storing tf record
    max_atoms : max. number of atoms to consider in a molecule.
    max_mass_spec_peak_loc : largest mass/charge ratio to allow in a spectra
    true_library_array_path_name: path for storing np.array of true spectra

  Returns:
    - Writes tf.Record of an example for each eligible molecule
    (i.e. # atoms < max_atoms)
    - Writes np.array (len(mol_list), max_mass_spec_peak_loc) to
      true_library_array_path_name if it is defined.
  """
  options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.ZLIB)

  # Wrapper function to add index value to dictionary
  if true_library_array_path_name:
    spectra_matrix = np.zeros((len(mol_list), max_mass_spec_peak_loc))

    def make_mol_dict_with_saved_array(idx, mol):
      mol_dict = make_mol_dict(mol, max_atoms, max_mass_spec_peak_loc)
      mol_dict[fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY] = idx
      spectra_matrix[idx, :] = mol_dict[fmap_constants.DENSE_MASS_SPEC]
      return mol_dict

    make_mol_dict_fn = make_mol_dict_with_saved_array

  else:

    def make_mol_dict_without_saved_array(idx, mol):
      del idx
      return make_mol_dict(mol, max_atoms, max_mass_spec_peak_loc)

    make_mol_dict_fn = make_mol_dict_without_saved_array

  with tf.python_io.TFRecordWriter(record_path_name, options) as writer:
    for idx, mol in enumerate(mol_list):
      mol_dict = make_mol_dict_fn(idx, mol)
      example = dict_to_tfexample(mol_dict)
      writer.write(example.SerializeToString())

  if true_library_array_path_name:
    with tf.gfile.Open(true_library_array_path_name, 'w') as f:
      np.save(f, spectra_matrix)


def write_info_file(mol_list, fname):
  """Write metadata for mol_list to fname.info."""

  num_elements = len(mol_list)
  with tf.gfile.Open(fname + '.info', 'w') as f:
    f.write('%d\n' % num_elements)


def parse_info_file(fname):
  """Parse dataset info from fname.info.

  The file format is very simple (but may be extended in the future).
  It is simply a text file containing a single line with an int describing
  the number of examples in the TFRecord file <fname>.

  Args:
    fname: input data file name
  Returns:
    a dictionary containing metadata about the data in fname
  """

  info_file = fname + '.info'

  info_dict = {}
  with tf.gfile.Open(info_file, 'r') as f:
    first_line = f.readline()
    info_dict['num_examples'] = int(first_line)

  return info_dict


def preprocess_spectrum(spectrum, hparams):
  if hparams.intensity_power != 1.0:
    return tf.pow(spectrum, hparams.intensity_power)
  else:
    return spectrum


def postprocess_spectrum(spectrum, hparams):
  if hparams.intensity_power != 1.0:
    return tf.pow(spectrum, 1. / hparams.intensity_power)
  else:
    return spectrum


def make_padded_shapes_for_dataset(dataset):
  """Makes a deep copy of dataset.output_shapes."""
  dataset_padded_shapes = {
      k: tf.TensorShape(v) for k, v in six.iteritems(dataset.output_shapes)
  }
  return dataset_padded_shapes


def _parse_example(example_protos, hparams, features_to_load):
  """Parsing map to create features for tf.Dataset.

  Args:
    example_protos: tf.Example proto read from TF.Records
    hparams: tf.HParams object, must contain
        max_atoms - Number of atoms in atom_weights array
        max_mass_spec_peak_loc - Number of bins in mass spectra
                                 Set to 2000 if unused.
    features_to_load: list of string keys of fields to load from the
      TFRecords. If None (default), all available fields are loaded.
  Returns:
    Dict containing functions for parsing features from a TFRecord.
  """
  features = {
      fmap_constants.MOLECULE_WEIGHT:
          tf.FixedLenFeature([1], tf.float32),
      fmap_constants.ATOM_WEIGHTS:
          tf.FixedLenFeature([hparams.max_atoms], tf.float32),
      fmap_constants.ATOM_IDS:
          tf.FixedLenFeature([hparams.max_atoms], tf.int64),
      fmap_constants.ADJACENCY_MATRIX:
          tf.FixedLenFeature([hparams.max_atoms * hparams.max_atoms], tf.int64),
      fmap_constants.DENSE_MASS_SPEC:
          tf.FixedLenFeature([hparams.max_mass_spec_peak_loc], tf.float32),
      fmap_constants.INCHIKEY:
          tf.FixedLenFeature([1], tf.string, default_value=''),
      fmap_constants.MOLECULAR_FORMULA:
          tf.FixedLenFeature([1], tf.string, default_value=''),
      fmap_constants.NAME:
          tf.FixedLenFeature([1], tf.string, default_value=''),
      fmap_constants.SMILES:
          tf.FixedLenFeature([1], tf.string, default_value=''),
      fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY:
          tf.FixedLenFeature([1], tf.int64, default_value=0),
      fmap_constants.SMILES_TOKEN_LIST_LENGTH:
          tf.FixedLenFeature([1], tf.int64, default_value=0)
  }

  for fp_len in ms_constants.NUM_CIRCULAR_FP_BITS_LIST:
    for rad in ms_constants.CIRCULAR_FP_RADII_LIST:
      for fp_type in fmap_constants.FP_TYPE_LIST:
        fp_key = ms_constants.CircularFingerprintKey(fp_type, fp_len, rad)
        features[str(fp_key)] = tf.FixedLenFeature([fp_key.fp_len], tf.float32)
  if features_to_load is not None:
    features = {key: features[key] for key in features_to_load}

  parsed_features = tf.parse_single_example(example_protos, features=features)

  if (features_to_load is None or
      fmap_constants.ADJACENCY_MATRIX in features_to_load):
    parsed_features[fmap_constants.ADJACENCY_MATRIX] = tf.reshape(
        parsed_features[fmap_constants.ADJACENCY_MATRIX],
        shape=(hparams.max_atoms, hparams.max_atoms))

  if (features_to_load is None or
      fmap_constants.DENSE_MASS_SPEC in features_to_load):
    parsed_features[fmap_constants.DENSE_MASS_SPEC] = preprocess_spectrum(
        parsed_features[fmap_constants.DENSE_MASS_SPEC], hparams)

  if (features_to_load is None or fmap_constants.SMILES in features_to_load):
    smiles_string = parsed_features[fmap_constants.SMILES]
    index_array = tf.py_func(feature_utils.tokenize_smiles,
                             [smiles_string], [tf.int64])
    index_array = tf.reshape(index_array, (-1,))
    parsed_features[fmap_constants.SMILES] = index_array
    parsed_features[fmap_constants.SMILES_TOKEN_LIST_LENGTH] = tf.shape(
        index_array)[0]
  return parsed_features


def get_dataset_in_one_batch(dataset, total_data_length):
  """Return all data in tf.Dataset in a single batch."""
  # Note that this line may raise some runtime warnings, since in general
  # composing .prefetch() and .cache() this way could be dropping data. However,
  # in our use case it is assumed that all of the data in the dataset is
  # contained in a single batch, so this order of caching and prefetching is
  # acceptable.
  dataset = dataset.prefetch(1).cache().repeat()

  # For downstream usages where we want the entire dataset in one batch we
  # also want the batch shape to be statically inferrable. Below, we
  # set that. Note that the only reason the set_shape command will fail
  # is if the size of the data is not what was provided in
  # data_info['num_examples'].
  def _set_static_batch_dimension(data):

    def _set_static_batch_dimension_for_tensor(tensor):
      shape = tensor.shape.as_list()
      shape[0] = total_data_length
      tensor.set_shape(shape)
      return tensor

    return tf.contrib.framework.nest.map_structure(
        _set_static_batch_dimension_for_tensor, data)

  return dataset.map(_set_static_batch_dimension)


def get_dataset_from_record(fnames,
                            hparams,
                            mode,
                            features_to_load=None,
                            all_data_in_one_batch=False):
  """Parse inputs from a tf.Record file containing molecular mass spec data.

  Args:
    fnames : list of names of TFRecord files. High level information about the
      the files will be read from each <fname>.info file.
    hparams: tf.HParams object, must contain
        max_atoms - Number of atoms in atom_weights array
        max_mass_spec_peak_loc - Number of bins in mass spectra
                                 Set to 2000 if unused.
        batch_size - Number of examples to return in a batch
        eval_batch_size - Number of examples to return in a batch during eval
    mode: Sets mode; if training mode, will shuffle dataset
    features_to_load: list of string keys of fields to load from the
      TFRecords. If None (default), all available fields are loaded.
    all_data_in_one_batch: whether all of the file's data should be placed
      in a single batch that repeats indefinitely.
  Returns:
    A tf.Dataset

  Raises:
    ValueError: if <fname>.info does not exist for any files in fnames.
  """

  if mode == tf.estimator.ModeKeys.TRAIN and len(fnames) > 1:
    # Throwing this warning because we do not interleave files.
    tf.logging.warn('Please ensure that shuffle buffer is large enough to'
                    ' effectively shuffle across all dataset input files.')

  if not fnames:
    raise ValueError('Input list of filenames is empty.')

  dataset = tf.data.TFRecordDataset(fnames, compression_type='ZLIB')

  total_data_length = sum(
      [parse_info_file(fname)['num_examples'] for fname in fnames])

  if mode == tf.estimator.ModeKeys.TRAIN and not all_data_in_one_batch:
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)

  # It is important to parse before we batch. Otherwise, the batched data
  # will have all of the fields in the input data, not just those in
  # features_to_load. This becomes very expensive if we are in
  # all_data_in_one_batch mode.
  dataset = dataset.map(
      lambda proto: _parse_example(proto, hparams, features_to_load))

  if all_data_in_one_batch:
    batch_size = total_data_length
  elif mode == tf.estimator.ModeKeys.TRAIN:
    batch_size = hparams.batch_size
  else:
    batch_size = hparams.eval_batch_size

  if features_to_load is None or fmap_constants.SMILES in features_to_load:
    padded_shapes = make_padded_shapes_for_dataset(dataset)
    padded_shapes[fmap_constants.SMILES] = ms_constants.MAX_TOKEN_LIST_LENGTH
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    assert dataset.output_shapes[fmap_constants.SMILES].ndims == 2

  else:
    dataset = dataset.batch(batch_size)

  if all_data_in_one_batch:
    dataset = get_dataset_in_one_batch(dataset, total_data_length)

  return dataset


def make_features_and_labels(dataset, feature_names, label_names, mode):
  """Get features and labels given a TFDataset.

  Args:
    dataset : TFDataset to parse
    feature_names : list of names of features, should be keys in dataset
    label_names : list of names of labels, should be keys in dataset
    mode : Sets either training or evaluation mode.

  Returns:
    features : Dictionary containing the subset of the input data corresponding
      to the feature_names and label_names. Note that we put *all* data in
      features, including prediction targets.
    labels : Dictionary containing a subset of the input data corresponding to
      label_names.

  Raises:
    ValueError : If any elements of feature_names or label_names are not keys
    in dataset.
  """

  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  if fmap_constants.SMILES in feature_names:
    feature_names.append(fmap_constants.SMILES_TOKEN_LIST_LENGTH)

  all_prop_names = set(feature_names + label_names)

  missing_props = tuple(p for p in all_prop_names if p not in next_element)

  if missing_props:
    raise ValueError('{} not found in TFRecord keys.'.format(missing_props))

  features = {n: next_element[n] for n in all_prop_names}
  labels = {n: next_element[n] for n in label_names}

  return features, labels


def load_training_spectra_array(spectra_array_path_name):
  """Get np.array stored in fpath.

  Assumes that the np.array is saved as mainlib_train_from_mainlib.npy, and
  is the np.array for the training data.

  Args:
    spectra_array_path_name: pathname to target np.array
  Returns:
    np.array with shape (len(training_data), ms_constants.max_peak_loc)
  """
  with tf.gfile.Open(spectra_array_path_name, 'rb') as f:
    spectra_array = np.load(f, encoding='bytes')

  return spectra_array


def validate_spectra_array_contents(record_path_name, hparams,
                                    spectra_array_path_name):
  """Checks that np.array containing spectra matches contents of record.

  Args:
    record_path_name: pathname to tf.Record file matching np.array
    hparams: See get_dataset_from_record
    spectra_array_path_name : pathname to spectra np.array.
  Raises:
    ValueError: if values in np.array stored at spectra_array_path_name
       does not match the spectra values in the TFRecord stored in the
       record_path_name.
  """
  dataset = get_dataset_from_record(
      [record_path_name],
      hparams,
      mode=tf.estimator.ModeKeys.EVAL,
      all_data_in_one_batch=True)

  feature_names = [fmap_constants.DENSE_MASS_SPEC]
  label_names = [fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY]

  features, labels = make_features_and_labels(
      dataset, feature_names, label_names, mode=tf.estimator.ModeKeys.EVAL)

  with tf.Session() as sess:
    feature_values, label_values = sess.run([features, labels])

  spectra_array = load_training_spectra_array(spectra_array_path_name)

  for i in range(np.shape(spectra_array)[0]):
    test_idx = label_values[fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY][i]
    spectra_from_dataset = feature_values[fmap_constants.DENSE_MASS_SPEC][
        test_idx, :]
    spectra_from_array = spectra_array[test_idx, :]

    if not all(spectra_from_dataset.flatten() == spectra_from_array.flatten()):
      raise ValueError('np.array of spectra stored at {} does not match spectra'
                       ' values in tf.Record {}'.format(spectra_array_path_name,
                                                        record_path_name))
  return
