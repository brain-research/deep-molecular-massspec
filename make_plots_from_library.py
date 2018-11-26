"""
Helper module for plotting spectral plots using existing prediction files and NIST sdf files. 
"""

import pandas as pd
import os
import glob
import seaborn as sns
import itertools
import numpy as np
import pickle as pkl

from rdkit import Chem

import parse_sdf_utils
import train_test_split_utils
import feature_utils
import mass_spec_constants as ms_constants
import plot_spectra_utils
import gather_similarities
from matplotlib import pyplot as plt

import pandas as pd
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole

from collections import OrderedDict
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string(
    'inchikey_file', None,
    'inchikey text file containing one inchikey per line.')


def load_model_results(prediction_file_pathname):
    '''Load predictions generated at end of training run from molecule_estimators.py.'''
    with open(prediction_file_pathname) as f:
        model_results = pd.read_csv(f, header=None, names=['query_ikey', 'match_ikey', 'rank'], delimiter=r"\s+", engine='python')
    return model_results 
  

def find_library_match_inchikey(query_inchikey, model_results_df):
    '''Find library matched spectra from DataFrame created by load_model_results.'''
    index = model_results_df['query_ikey'].values.tolist().index(query_inchikey)
    matched_inchikey = model_results_df.iloc[index]['match_ikey']
    return matched_inchikey

 
def plot_mainlib_and_replicate_spectra(inchikey, inchikey_dict, replicates_inchikey_dict, image_dir=''):
    replicates_spectra = gather_similarities.make_spectra_array_from_mol_list(replicates_inchikey_dict[inchikey])
    mainlib_spectra = gather_similarities.make_spectra_array_from_mol_list(inchikey_dict[inchikey])

    if image_dir:
        image_pathname = os.path.join(image_dir, '{}_mainlib_replicates.png'.format(inchikey))
        
        plot_spectra_utils.plot_true_and_predicted_spectra(
            mainlib_spectra[0,:],
            replicates_spectra[0,:],
            plot_mode_key=plot_spectra_utils.PlotModeKeys.MAINLIB_REPLICATES_SPECTRUM,
            output_filename=image_pathname,
            rescale_mz_axis=True)
    return 


def plot_predicted_and_mainlib_spectra(inchikey, inchikey_dict, predicted_spectra_dict, image_dir=''):
    true_spectra = gather_similarities.make_spectra_array_from_mol_list(inchikey_dict[inchikey])  
    predicted_spectra = np.reshape(predicted_spectra_dict[inchikey], (1, 1000)) 
    predicted_spectra = predicted_spectra / np.amax(predicted_spectra, axis=1, keepdims=True) * 1000 
    
    if image_dir:
        image_pathname = os.path.join(image_dir, '{}_mainlib_predicted.png'.format(inchikey))

        plot_spectra_utils.plot_true_and_predicted_spectra(
            true_spectra[0,:],
            predicted_spectra[0,:],
            plot_mode_key=plot_spectra_utils.PlotModeKeys.PREDICTED_SPECTRUM,
            output_filename=image_pathname,
            rescale_mz_axis=True,
            large_tick_size=True)
    return 
 
 
def plot_query_and_library_match_spectra(
    inchikey, model_results_df, mainlib_inchikey_dict, 
    replicates_inchikey_dict, predicted_spectra_dict, image_dir=''):
    '''Plot query and library match spectra.

    Args:
      inchikey: inchikey to analyze
      model_results_df: a pd.Dataframe loaded from a predictions file.
      mainlib_inchikey_dict: Collection of main library mol objects, keyed by inchikey
      replicates_inchikey_dict: Collection of replicates mol objects, keyed by inchikey
      predicted_spectra_dict: Collection of spectral predictions keyed by inchikey; should correspond to those used in library matching
    Output:
      Spectral image comparing the query spectra and the library matched spectra.
    '''
    lib_match_ikey = find_library_match_inchikey(inchikey, model_results_df)
    query_spectra_replicates = gather_similarities.make_spectra_array_from_mol_list(replicates_inchikey_dict[inchikey])

    if lib_match_ikey in predicted_spectra_dict: 
        lib_match_spectra = np.reshape(predicted_spectra_dict[lib_match_ikey], (1, 1000)) 
        lib_match_spectra = lib_match_spectra / np.amax(lib_match_spectra, axis=1, keepdims=True) * 1000 
    else:
        lib_match_spectra = gather_similarities.make_spectra_array_from_mol_list(mainlib_inchikey_dict[lib_match_ikey])
    
    if image_dir:
        image_pathname = os.path.join(image_dir, '{}_matched_to_{}.png'.format(inchikey, lib_match_ikey))
        
        plot_spectra_utils.plot_true_and_predicted_spectra(
            query_spectra_replicates[0,:],
            lib_match_spectra[0,:],
            plot_mode_key=plot_spectra_utils.PlotModeKeys.LIBRARY_MATCHED_SPECTRUM,
            output_filename=image_pathname,
            rescale_mz_axis=True)
    return


def main():
  # preprocess data
  mol_list_17 = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/mainlib_mend.sdf')
  inchikey_dict = train_test_split_utils.make_inchikey_dict(mol_list_17)
  replicates_mol_list_17 = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/replib_mend.sdf')
  replicates_inchikey_dict = train_test_split_utils.make_inchikey_dict(replicates_mol_list_17)

  # grab library match prediction results
  model_results = load_model_results('/mnt/storage/massspec_results/results_9_19/mlp_bidirectional_no_mass_filter.test.filter_False/95819.library_matching_predictions.txt')
  
  predicted_spectra_dict = np.load('/tmp/spectra_array_predictions/mlp_bidirectional_predictions/no_family_validation.npy').item()
  predicted_spectra_dict.update(np.load('/tmp/spectra_array_predictions/mlp_bidirectional_predictions/fentanyl_validation.npy').item())
  predicted_spectra_dict.update(np.load('/tmp/spectra_array_predictions/mlp_bidirectional_predictions/steroid_validation.npy').item())
  predicted_spectra_dict.update(np.load('/tmp/spectra_array_predictions/mlp_bidirectional_predictions/no_family_test.npy').item())
  predicted_spectra_dict.update(np.load('/tmp/spectra_array_predictions/mlp_bidirectional_predictions/fentanyl_test.npy').item())
  predicted_spectra_dict.update(np.load('/tmp/spectra_array_predictions/mlp_bidirectional_predictions/steroid_test.npy').item())

  # test_ikey = predicted_spectra_dict.keys()[0] 
  with open(FLAGS.inchikey_file) as f:
    inchikeys = [l.strip() for l in f.readlines()]
  

  base_dir, fname = os.path.split(FLAGS.inchikey_file) 
  base_fname = fname.split('.')[0]
  image_dir = os.path.join(base_dir, base_fname+'_spectra')

  if not os.path.exists(image_dir):
    os.mkdir(image_dir)

  for key in inchikeys:
    # plot_mainlib_and_replicate_spectra(key, inchikey_dict, replicates_inchikey_dict, image_dir)
    plot_predicted_and_mainlib_spectra(key, inchikey_dict, predicted_spectra_dict, image_dir)
    # plot_query_and_library_match_spectra(key, model_results, inchikey_dict, replicates_inchikey_dict, predicted_spectra_dict, image_dir)	

if __name__ == '__main__':
	main()
