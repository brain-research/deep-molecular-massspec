from rdkit import Chem
import numpy as np
import tensorflow as tf
import pickle as pkl
import pandas as pd

import sys
import os
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

# load predicted spectra dict
predicted_spectra_dict = np.load('/tmp/mlp_bidirectional_predictions/no_family_test.npy').item()
predicted_spectra_dict.update(np.load('/tmp/mlp_bidirectional_predictions/fentanyl_test.npy').item())
predicted_spectra_dict.update(np.load('/tmp/mlp_bidirectional_predictions/steroid_test.npy').item())
predicted_spectra_dict.update(np.load('/tmp/mlp_bidirectional_predictions/no_family_validation.npy').item())
predicted_spectra_dict.update(np.load('/tmp/mlp_bidirectional_predictions/fentanyl_validation.npy').item())
predicted_spectra_dict.update(np.load('/tmp/mlp_bidirectional_predictions/steroid_validation.npy').item())


def is_inchikey_valid(inchikey):
    return inchikey in mainlib_inchikey_dict

def make_save_similarities(ikey_list, result_base_name='/tmp/similarity_results/', shard_idx=0, total_shard_number=0):
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
 
    predicted_spectra = np.array([predicted_spectra_dict[ikey] for ikey in ikey_list])
    mainlib_spectra = np.array([gather_similarities.make_spectra_array_from_mol(mainlib_inchikey_dict[ikey][0]) for ikey in ikey_list])
    replib_spectra = np.array([gather_similarities.make_spectra_array_from_mol(replicates_inchikey_dict[ikey][0]) for ikey in ikey_list])
    library_match_spectra = np.array(library_match_spectra)
    
    mainlib_predicted_similarities = gather_similarities.get_similarity_two_spectra_sets(mainlib_spectra, predicted_spectra)
    np.save(name_output_file('mainlib_predicted_similarities'), mainlib_predicted_similarities)

    replicate_library_match_similarities = gather_similarities.get_similarity_two_spectra_sets(replib_spectra, library_match_spectra) 
    np.save(name_output_file('replicate_library_match_similarities'), replicate_library_match_similarities)
     

if __name__ == '__main__':
    model_results = make_plots_from_library.load_model_results('/mnt/storage/massspec_results/results_9_19/mlp_bidirectional_no_mass_filter.test.filter_False/95819.library_matching_predictions.txt')

    results_dir = '/mnt/storage/massspec_misc/all_replicates_similarities/'

    shard_size = 1000
    total_shards = len(model_results)/shard_size
    for i in range(6, 7):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size
        if end_idx > len(model_results):
            end_idx = -1
        make_save_similarities(model_results['query_ikey'][start_idx : end_idx].tolist(), results_dir, i, total_shards) 
