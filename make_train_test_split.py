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

r"""Creates datasets from the NIST sdf files and makes experiment setup jsons.

This module first breaks up the main NIST library dataset into a train/
validation/test set, and the replicates library into a validation and test set.
As all the molecules in the replicates file are also in the main NIST library,
the mainlib datasets will exclude inchikeys from the replicates library. All the
molecules in both datasets are to be included in one of these datasets, unless
an argument is passed for mainlib_maximum_num_molecules_to_use or
replicates_maximum_num_molecules_to_use.

The component datasets are saved as TFRecords, by the names defined in
dataset_setup_constants and the library from which the data came
(e.g. mainlib_train_from_mainlib.tfrecord). This will result in 7 TFRecord files
total, one each for the train/validation/test splits from the main library, and
two each for the replicates validation/test splits, one with its data from the
mainlib NIST file, and the other from the replicates file.

For each experiment setup included in
dataset_setup_constants.EXPERIMENT_SETUPS_LIST, a json file is written. This
json file name the files to be used for each part of the experiment, i.e.
library matching, spectra prediction.

Note: Reading sdf files from cns currently not supported.

Example usage:
make_train_test_split.py \
--main_sdf_name=testdata/test_14_mend.sdf
--replicates_sdf_name=testdata/test_2_mend.sdf \
--output_master_dir=<output_dir_name>

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import random

from absl import app
from absl import flags
import dataset_setup_constants as ds_constants
import mass_spec_constants as ms_constants
import parse_sdf_utils
import train_test_split_utils
import six
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'main_sdf_name', 'testdata/test_14_mend.sdf',
    'specify full path of sdf file to parse, to be used for'
    ' training sets, and validation/test sets')
flags.DEFINE_string(
    'replicates_sdf_name',
    'testdata/test_2_mend.sdf',
    'specify full path of a second sdf file to parse, to be'
    ' used for the vaildation/test set. Molecules in this sdf'
    ' will be excluded from the main train/val/test sets.')
# Note: For family based splitting, all molecules passing the filter will be
#  placed in validation/test datasets, and then split according to the relative
#  ratio between the validation/test fractions. If these are both equal to 0.0,
#  these values will be over written to 0.5 and 0.5.
flags.DEFINE_list(
    'main_train_val_test_fractions', '1.0,0.0,0.0',
    'specify how large to make the train, val, and test sets'
    ' as a fraction of the whole dataset.')
flags.DEFINE_integer('mainlib_maximum_num_molecules_to_use', None,
                     'specify how many total samples to use for parsing')
flags.DEFINE_integer('replicates_maximum_num_molecules_to_use', None,
                     'specify how many total samples to use for parsing')
flags.DEFINE_list(
    'replicates_train_val_test_fractions', '0.0,0.5,0.5',
    'specify fraction of replicates molecules to use for'
    ' for the three replicates sample files.')
flags.DEFINE_enum(
    'splitting_type', 'random', ['random', 'steroid', 'diazo'],
    'specify splitting method to use for creating '
    'training/validation/test sets')
flags.DEFINE_string('output_master_dir', '/tmp/output_dataset_dir',
                    'specify directory to save records')
flags.DEFINE_integer('max_atoms', ms_constants.MAX_ATOMS,
                     'specify maximum number of atoms to allow')
flags.DEFINE_integer('max_mass_spec_peak_loc', ms_constants.MAX_PEAK_LOC,
                     'specify greatest m/z spectrum peak to allow')

INCHIKEY_FILENAME_END = '.inchikey.txt'
TFRECORD_FILENAME_END = '.tfrecord'
NP_LIBRARY_ARRAY_END = '.spectra_library.npy'
FROM_MAINLIB_FILENAME_MODIFIER = '_from_mainlib'
FROM_REPLICATES_FILENAME_MODIFIER = '_from_replicates'


def make_mainlib_replicates_train_test_split(
    mainlib_mol_list,
    replicates_mol_list,
    splitting_type,
    mainlib_fractions,
    replicates_fractions,
    mainlib_maximum_num_molecules_to_use=None,
    replicates_maximum_num_molecules_to_use=None,
    rseed=42):
  """Makes train/validation/test inchikey lists from two lists of rdkit.Mol.

  Args:
    mainlib_mol_list : list of molecules from main library
    replicates_mol_list : list of molecules from replicates library
    splitting_type : type of splitting to use for validation splits.
    mainlib_fractions : TrainValTestFractions namedtuple
        holding desired fractions for train/val/test split of mainlib
    replicates_fractions : TrainValTestFractions namedtuple
        holding desired fractions for train/val/test split of replicates.
        For the replicates set, the train fraction should be set to 0.
    mainlib_maximum_num_molecules_to_use : Largest number of molecules to use
       when making datasets from mainlib
    replicates_maximum_num_molecules_to_use : Largest number of molecules to use
       when making datasets from replicates
    rseed : random seed for shuffling

  Returns:
    main_inchikey_dict : Dict that is keyed by inchikey, containing a list of
        rdkit.Mol objects corresponding to that inchikey from the mainlib
    replicates_inchikey_dict : Dict that is keyed by inchikey, containing a list
        of rdkit.Mol objects corresponding to that inchikey from the replicates
        library
    main_replicates_split_inchikey_lists_dict : dict with keys :
      'mainlib_train', 'mainlib_validation', 'mainlib_test',
      'replicates_train', 'replicates_validation', 'replicates_test'
      Values are lists of inchikeys corresponding to each dataset.

  """
  random.seed(rseed)
  main_inchikey_dict = train_test_split_utils.make_inchikey_dict(
      mainlib_mol_list)
  main_inchikey_list = main_inchikey_dict.keys()

  if six.PY3:
    main_inchikey_list = list(main_inchikey_list)

  if mainlib_maximum_num_molecules_to_use is not None:
    main_inchikey_list = random.sample(main_inchikey_list,
                                       mainlib_maximum_num_molecules_to_use)

  replicates_inchikey_dict = train_test_split_utils.make_inchikey_dict(
      replicates_mol_list)
  replicates_inchikey_list = replicates_inchikey_dict.keys()

  if six.PY3:
    replicates_inchikey_list = list(replicates_inchikey_list)

  if replicates_maximum_num_molecules_to_use is not None:
    replicates_inchikey_list = random.sample(
        replicates_inchikey_list, replicates_maximum_num_molecules_to_use)

  # Make train/val/test splits for main dataset.
  main_train_validation_test_inchikeys = (
      train_test_split_utils.make_train_val_test_split_inchikey_lists(
          main_inchikey_list,
          main_inchikey_dict,
          mainlib_fractions,
          holdout_inchikey_list=replicates_inchikey_list,
          splitting_type=splitting_type))

  # Make train/val/test splits for replicates dataset.
  replicates_validation_test_inchikeys = (
      train_test_split_utils.make_train_val_test_split_inchikey_lists(
          replicates_inchikey_list,
          replicates_inchikey_dict,
          replicates_fractions,
          splitting_type=splitting_type))

  component_inchikey_dict = {
      ds_constants.MAINLIB_TRAIN_BASENAME:
          main_train_validation_test_inchikeys.train,
      ds_constants.MAINLIB_VALIDATION_BASENAME:
          main_train_validation_test_inchikeys.validation,
      ds_constants.MAINLIB_TEST_BASENAME:
          main_train_validation_test_inchikeys.test,
      ds_constants.REPLICATES_TRAIN_BASENAME:
          replicates_validation_test_inchikeys.train,
      ds_constants.REPLICATES_VALIDATION_BASENAME:
          replicates_validation_test_inchikeys.validation,
      ds_constants.REPLICATES_TEST_BASENAME:
          replicates_validation_test_inchikeys.test
  }

  train_test_split_utils.assert_all_lists_mutally_exclusive(
      list(component_inchikey_dict.values()))
  # Test that the set of the 5 component inchikey lists is equal to the set of
  #   inchikeys in the main library.
  all_inchikeys_in_components = []
  for ikey_list in list(component_inchikey_dict.values()):
    for ikey in ikey_list:
      all_inchikeys_in_components.append(ikey)

  assert set(main_inchikey_list + replicates_inchikey_list) == set(
      all_inchikeys_in_components
  ), ('The inchikeys in the original inchikey dictionary are not all included'
      ' in the train/val/test component libraries')

  return (main_inchikey_dict, replicates_inchikey_dict, component_inchikey_dict)


def write_list_of_inchikeys(inchikey_list, base_name, output_dir):
  """Write list of inchikeys as a text file."""
  inchikey_list_name = base_name + INCHIKEY_FILENAME_END

  with tf.gfile.Open(os.path.join(output_dir, inchikey_list_name),
                     'w') as writer:
    for inchikey in inchikey_list:
      writer.write('%s\n' % inchikey)


def write_all_dataset_files(inchikey_dict,
                            inchikey_list,
                            base_name,
                            output_dir,
                            max_atoms,
                            max_mass_spec_peak_loc,
                            make_library_array=False):
  """Helper function for writing all the files associated with a TFRecord.

  Args:
    inchikey_dict : Full dictionary keyed by inchikey containing lists of
                    rdkit.Mol objects
    inchikey_list : List of inchikeys to include in dataset
    base_name : Base name for the dataset
    output_dir : Path for saving all TFRecord files
    max_atoms : Maximum number of atoms to include for a given molecule
    max_mass_spec_peak_loc : Largest m/z peak to include in a spectra.
    make_library_array : Flag for whether to make library array
  Returns:
    Saves 3 files:
     basename.tfrecord : a TFRecord file,
     basename.inchikey.txt : a text file with all the inchikeys in the dataset
     basename.tfrecord.info: a text file with one line describing
         the length of the TFRecord file.
    Also saves if make_library_array is set:
     basename.npz : see parse_sdf_utils.write_dicts_to_example
  """
  record_name = base_name + TFRECORD_FILENAME_END

  mol_list = train_test_split_utils.make_mol_list_from_inchikey_dict(
      inchikey_dict, inchikey_list)

  if make_library_array:
    library_array_pathname = base_name + NP_LIBRARY_ARRAY_END
    parse_sdf_utils.write_dicts_to_example(
        mol_list, os.path.join(output_dir, record_name),
        max_atoms, max_mass_spec_peak_loc,
        os.path.join(output_dir, library_array_pathname))
  else:
    parse_sdf_utils.write_dicts_to_example(
        mol_list, os.path.join(output_dir, record_name), max_atoms,
        max_mass_spec_peak_loc)
  write_list_of_inchikeys(inchikey_list, base_name, output_dir)
  parse_sdf_utils.write_info_file(mol_list, os.path.join(
      output_dir, record_name))


def write_mainlib_split_datasets(component_inchikey_dict, mainlib_inchikey_dict,
                                 output_dir, max_atoms, max_mass_spec_peak_loc):
  """Write all train/val/test set TFRecords from main NIST sdf file."""
  for component_kwarg in component_inchikey_dict.keys():
    component_mainlib_filename = (
        component_kwarg + FROM_MAINLIB_FILENAME_MODIFIER)
    if component_kwarg == ds_constants.MAINLIB_TRAIN_BASENAME:
      write_all_dataset_files(
          mainlib_inchikey_dict,
          component_inchikey_dict[component_kwarg],
          component_mainlib_filename,
          output_dir,
          max_atoms,
          max_mass_spec_peak_loc,
          make_library_array=True)
    else:
      write_all_dataset_files(mainlib_inchikey_dict,
                              component_inchikey_dict[component_kwarg],
                              component_mainlib_filename, output_dir, max_atoms,
                              max_mass_spec_peak_loc)


def write_replicates_split_datasets(component_inchikey_dict,
                                    replicates_inchikey_dict, output_dir,
                                    max_atoms, max_mass_spec_peak_loc):
  """Write replicates val/test set TFRecords from replicates sdf file."""
  for component_kwarg in [
      ds_constants.REPLICATES_VALIDATION_BASENAME,
      ds_constants.REPLICATES_TEST_BASENAME
  ]:
    component_replicates_filename = (
        component_kwarg + FROM_REPLICATES_FILENAME_MODIFIER)
    write_all_dataset_files(replicates_inchikey_dict,
                            component_inchikey_dict[component_kwarg],
                            component_replicates_filename, output_dir,
                            max_atoms, max_mass_spec_peak_loc)


def combine_inchikey_sets(dataset_subdivision_list, dataset_split_dict):
  """A function to combine lists of inchikeys that are values from a dict.

  Args:
    dataset_subdivision_list: List of keys in dataset_split_dict to combine
        into one list
    dataset_split_dict: dict containing keys in dataset_subdivision_list, with
        lists of inchikeys as values.
  Returns:
    A list of inchikeys.
  """
  dataset_inchikey_list = []
  for dataset_subdivision_name in dataset_subdivision_list:
    dataset_inchikey_list.extend(dataset_split_dict[dataset_subdivision_name])
  return dataset_inchikey_list


def check_experiment_setup(experiment_setup_dict, component_inchikey_dict):
  """Validates experiment setup for given lists of inchikeys."""

  # Check that the union of the library matching observed and library
  #   matching predicted sets are equal to the set of inchikeys in the
  #   mainlib_inchikey_dict
  all_inchikeys_in_library = (
      combine_inchikey_sets(
          experiment_setup_dict[ds_constants.LIBRARY_MATCHING_OBSERVED_KEY],
          component_inchikey_dict) +
      combine_inchikey_sets(
          experiment_setup_dict[ds_constants.LIBRARY_MATCHING_PREDICTED_KEY],
          component_inchikey_dict))

  all_inchikeys_in_use = []
  for kwarg in component_inchikey_dict.keys():
    all_inchikeys_in_use.extend(component_inchikey_dict[kwarg])

  assert set(all_inchikeys_in_use) == set(all_inchikeys_in_library), (
      'Inchikeys in library for library matching does not match full dataset.')

  # Check that all inchikeys in query are found in full library of inchikeys.
  assert set(
      combine_inchikey_sets(
          experiment_setup_dict[ds_constants.LIBRARY_MATCHING_QUERY_KEY],
          component_inchikey_dict)).issubset(set(all_inchikeys_in_library)), (
              'Inchikeys in query set for library matching not'
              'found in library.')


def write_json_for_experiment(experiment_setup, output_dir):
  """Writes json for experiment, recording relevant files for each component.

  Writes a json containing a list of TFRecord file names to read
  for each experiment component, i.e. spectrum_prediction, library_matching.

  Args:
    experiment_setup: A dataset_setup_constants.ExperimentSetup tuple
    output_dir: directory to write json
  Returns:
    Writes json recording which files to load for each component
    of the experiment
  Raises:
    ValueError: if the experiment component is not specified to be taken from
        either the main NIST library or the replicates library.

  """
  experiment_json_dict = {}
  for dataset_kwarg in experiment_setup.experiment_setup_dataset_dict:
    if dataset_kwarg in experiment_setup.data_to_get_from_mainlib:
      experiment_json_dict[dataset_kwarg] = [
          (component_basename + FROM_MAINLIB_FILENAME_MODIFIER +
           TFRECORD_FILENAME_END) for component_basename in
          experiment_setup.experiment_setup_dataset_dict[dataset_kwarg]
      ]
    elif dataset_kwarg in experiment_setup.data_to_get_from_replicates:
      experiment_json_dict[dataset_kwarg] = [
          (component_basename + FROM_REPLICATES_FILENAME_MODIFIER +
           TFRECORD_FILENAME_END) for component_basename in
          experiment_setup.experiment_setup_dataset_dict[dataset_kwarg]
      ]
    else:
      raise ValueError('Did not specify origin for {}.'.format(dataset_kwarg))

  training_spectra_filename = (
      ds_constants.MAINLIB_TRAIN_BASENAME + FROM_MAINLIB_FILENAME_MODIFIER +
      NP_LIBRARY_ARRAY_END)
  experiment_json_dict[
      ds_constants.TRAINING_SPECTRA_ARRAY_KEY] = training_spectra_filename

  with tf.gfile.Open(os.path.join(output_dir, experiment_setup.json_name),
                     'w') as writer:
    experiment_json = json.dumps(experiment_json_dict)
    writer.write(experiment_json)


def main(_):
  tf.gfile.MkDir(FLAGS.output_master_dir)

  main_train_val_test_fractions_tuple = tuple(
      [float(elem) for elem in FLAGS.main_train_val_test_fractions])
  main_train_val_test_fractions = train_test_split_utils.TrainValTestFractions(
      *main_train_val_test_fractions_tuple)

  replicates_train_val_test_fractions_tuple = tuple(
      [float(elem) for elem in FLAGS.replicates_train_val_test_fractions])
  replicates_train_val_test_fractions = (
      train_test_split_utils.TrainValTestFractions(
          *replicates_train_val_test_fractions_tuple))

  mainlib_mol_list = parse_sdf_utils.get_sdf_to_mol(
      FLAGS.main_sdf_name, max_atoms=FLAGS.max_atoms)
  replicates_mol_list = parse_sdf_utils.get_sdf_to_mol(
      FLAGS.replicates_sdf_name, max_atoms=FLAGS.max_atoms)

  # Breaks the inchikeys lists into train/validation/test splits.
  (mainlib_inchikey_dict, replicates_inchikey_dict, component_inchikey_dict) = (
      make_mainlib_replicates_train_test_split(
          mainlib_mol_list,
          replicates_mol_list,
          FLAGS.splitting_type,
          main_train_val_test_fractions,
          replicates_train_val_test_fractions,
          mainlib_maximum_num_molecules_to_use=FLAGS.
          mainlib_maximum_num_molecules_to_use,
          replicates_maximum_num_molecules_to_use=FLAGS.
          replicates_maximum_num_molecules_to_use))

  # Writes TFRecords for each component using info from the main library file
  write_mainlib_split_datasets(component_inchikey_dict, mainlib_inchikey_dict,
                               FLAGS.output_master_dir, FLAGS.max_atoms,
                               FLAGS.max_mass_spec_peak_loc)

  # Writes TFRecords for each component using info from the replicates file
  write_replicates_split_datasets(
      component_inchikey_dict, replicates_inchikey_dict,
      FLAGS.output_master_dir, FLAGS.max_atoms, FLAGS.max_mass_spec_peak_loc)

  for experiment_setup in ds_constants.EXPERIMENT_SETUPS_LIST:
    # Check that experiment setup is valid.
    check_experiment_setup(experiment_setup.experiment_setup_dataset_dict,
                           component_inchikey_dict)

    # Write a json for the experiment setups, pointing to local files.
    write_json_for_experiment(experiment_setup, FLAGS.output_master_dir)


if __name__ == '__main__':
  app.run(main)
