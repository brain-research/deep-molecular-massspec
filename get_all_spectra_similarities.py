from rdkit import Chem
import numpy as np
import tensorflow as tf
import pickle as pkl
import pandas as pd

import sys
import os
import glob
sys.path.append('/home/jennifer/git-files/deep-molecular-massspec/')

import parse_sdf_utils
import train_test_split_utils
import feature_utils
import mass_spec_constants as ms_constants
import molecule_predictors
import similarity as similarity_lib
import gather_similarities
import make_plots_from_library

mol_list_17 = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/mainlib_mend.sdf')
mainlib_inchikey_dict = train_test_split_utils.make_inchikey_dict(mol_list_17)

replicates_mol_list_17 = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/replib_mend.sdf')
replicates_inchikey_dict = train_test_split_utils.make_inchikey_dict(replicates_mol_list_17)

def get_predictions_dict(path):
  """Load a set of predictions as a dictionary given a path.
  Args:
    path: path containing spectra predictions as np.array, with results stored as dictionary
  Output:
    dict containing spectra predictions keyed by inchikey
  """
  prediction_filenames = glob.glob(path+'/*') 
  predicted_spectra_dict = {}
  for fname in prediction_filenames: 
    predicted_spectra_dict.update(np.load(fname).item())
  return predicted_spectra_dict


def is_inchikey_valid(inchikey):
    return inchikey in mainlib_inchikey_dict

def make_save_similarities(ikey_list, result_base_name='/tmp/similarity_results/', predictions_dir='/tmp/spectra_array_predictions/mlp_bidirectional_prediction', shard_idx=0, total_shard_number=0):
    """Save similarity files for a list of inchikeys.
    Args:
      ikey_list: list of inchikeys to calculate the similarity of
      result_base_name: Directory for storing results files
      predictions_dir: Directory containing spectra predictions. Passed to get_predictions_dict
      shard_idx: Shard number for use when recording shard number
      total_shard_number: Total number of shards to make. Set to 0 to turn off this labeling
    Output:
      Saves a np.array for the following values. Each should have len(ikey_list) rows
        inchikeys: List of inchikeys corresponding to other rows
        mainlib_replib_similarities: Similarity measurment between mainlib spectrum 
            and replicates spectra
        overall_similarities: Overall similarities between all the spectra 
            (including replicates to replicates)
        mainlib_predicted_similarities: Similarity between mainlib spectrum and predicted spectrum
        replicate_library_match_similarities: Similarity between the query spectrum from replicates
            and the top ranked library matched spectrum

    """
    mainlib_replib_similarities = []
    overall_similarities = []
    library_match_spectra = []
    ranks = []
    bad_inchikeys = []

    def name_output_file(name):
        if total_shard_number:
            return result_base_name + '{}_of_{}_{}.npy'.format(shard_idx+1, total_shard_number, name)
        else:
            return result_base_name + '{}.npy'.format(name)


    for idx, ikey in enumerate(ikey_list):
        if idx % 1000 == 0:
            print idx

        library_match_ikey = make_plots_from_library.find_library_match_inchikey(ikey, model_results)
        if not is_inchikey_valid(ikey): 
            tf.logging.warn('{} not found, skipping molecule'.format(ikey))
            bad_inchikeys.append(ikey)
            continue
        elif not is_inchikey_valid(library_match_ikey):
            tf.logging.warn('{} not found, skipping molecule'.format(library_match_ikey))
            bad_inchikeys.append(ikey)
            continue

        # For the inter library analysis, we want to use all of the spectra
        # unlike for the comparisons against the predicted library.
        mainlib_spectra = gather_similarities.make_spectra_array_from_mol_list(mainlib_inchikey_dict[ikey])
        replib_spectra = gather_similarities.make_spectra_array_from_mol_list(replicates_inchikey_dict[ikey])

        mainlib_sim, overall_sim = gather_similarities.get_mainlib_replib_similarities(mainlib_spectra[0, :], replib_spectra)
        mainlib_replib_similarities.append(mainlib_sim)
        overall_similarities.append(overall_sim)
        
        idx = model_results['query_ikey'].tolist().index(ikey)
        ranks.append(int(model_results.iloc[idx]['rank']) + 1)
        
        if library_match_ikey in predicted_spectra_dict:
            # if library_match_ikey == ikey:
            #     print 'same ikey for library match'
            library_match_spectra.append(predicted_spectra_dict[ikey])
        else:
            library_match_spectra.append(
                gather_similarities.make_spectra_array_from_mol(mainlib_inchikey_dict[library_match_ikey][0]))
        

    for bad_ikey in bad_inchikeys: 
        ikey_list.remove(bad_ikey)

    if len(ikey_list) != len(library_match_spectra):
        raise ValueError('Bad inchikeys not properly removed from the file.')

    np.save(name_output_file('inchikey'), ikey_list)
    np.save(name_output_file('mainlib_replib_similarities'), mainlib_replib_similarities)
    np.save(name_output_file('overall_similarities'), overall_similarities)
 
    predicted_spectra_dict = get_predictions_dict(predictions_dir) 
    predicted_spectra = np.array([predicted_spectra_dict[ikey] for ikey in ikey_list])
    mainlib_spectra = np.array([gather_similarities.make_spectra_array_from_mol(mainlib_inchikey_dict[ikey][0]) for ikey in ikey_list])
    replib_spectra = np.array([gather_similarities.make_spectra_array_from_mol(replicates_inchikey_dict[ikey][0]) for ikey in ikey_list])
    library_match_spectra = np.array(library_match_spectra)
    
    mainlib_predicted_similarities = gather_similarities.get_similarity_two_spectra_sets(mainlib_spectra, predicted_spectra)
    np.save(name_output_file('mainlib_predicted_similarities'), mainlib_predicted_similarities)

    replicate_library_match_similarities = gather_similarities.get_similarity_two_spectra_sets(replib_spectra, library_match_spectra) 
    np.save(name_output_file('replicate_library_match_similarities'), replicate_library_match_similarities)
 

def get_predicted_similarities(ikey_list, prediction_dir, model_name, result_basename='/tmp/similarity_results/'):
  """Gets predicted for molecules in the inchikey list."""
  predicted_spectra_dict = get_predictions_dict(prediction_dir)
  predicted_spectra = np.array([predicted_spectra_dict[ikey] for ikey in ikey_list])
  mainlib_spectra = np.array([gather_similarities.make_spectra_array_from_mol(mainlib_inchikey_dict[ikey][0]) for ikey in ikey_list])
  mainlib_predicted_similarities = gather_similarities.get_similarity_two_spectra_sets(mainlib_spectra, predicted_spectra)
  np.save(result_basename + model_name, mainlib_predicted_similarities)


def run_all_predicted_results(ikey_list): 

    results_dir = '/mnt/storage/massspec_misc/all_replicates_similarities/'

    shard_size = 1000
    total_shards = len(model_results)/shard_size
    for i in range(6, 7):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size
        if end_idx > len(model_results):
            end_idx = -1
        make_save_similarities(ikey_list[start_idx : end_idx], results_dir, i, total_shards) 


if __name__ == '__main__':
    model_results = make_plots_from_library.load_model_results('/mnt/storage/massspec_results/results_9_19/mlp_bidirectional_no_mass_filter.test.filter_False/95819.library_matching_predictions.txt')
    ikey_list = model_results['query_ikey'].tolist()

    get_predicted_similarities(ikey_list, '/tmp/spectra_array_predictions/mlp_forward_predictions', 'mlp_forward_prediction', '/mnt/storage/massspec_misc/predicted_similarities/' )
    # run_all_predicted_results(ikey_list)
