from rdkit import Chem
import numpy as np
import tensorflow as tf
import pickle as pkl

import parse_sdf_utils
import train_test_split_utils
import feature_utils
import mass_spec_constants as ms_constants
import similarity as similarity_lib


def make_spectra_array_from_mol(mol):
    """Get spectra array from Rdkit.Mol object."""
    spectra_str = mol.GetProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS)
    spectral_locs, spectral_intensities = feature_utils.parse_peaks(spectra_str)
    dense_mass_spec = feature_utils.make_dense_mass_spectra(
        spectral_locs, spectral_intensities, ms_constants.MAX_PEAK_LOC)
    return dense_mass_spec


def make_spectra_array_from_mol_list(mol_list):
    """Grab spectra pertaining to same molecule in one np.array.
    Args:
      mol_list: list of rdkit.Mol objects. Each Mol should contain 
          information about the spectra, as stored in NIST.
    Output: 
      np.array of spectra of shape (number of spectra, max spectra length)
    """
    mass_spec_spectra = np.zeros( ( len(mol_list), ms_constants.MAX_PEAK_LOC))
    for idx, mol in enumerate(mol_list):
        dense_mass_spec = make_spectra_array_from_mol(mol) 
        mass_spec_spectra[idx, :] = dense_mass_spec
    
    return mass_spec_spectra


def get_similarities(raw_spectra_array):
    """Preprocess spectra and then calculate similarity between spectra.
    Args:
        raw_spectra_array: np.array containing unprocessed spectra
    Output:
        np.array of shape (len(raw_spectra_array), len(raw_spectra_array))
        reflects distances between spectra.
    """
    spec_array_var = tf.constant(raw_spectra_array)

    # Adjusting intensity to match default in molecule_predictors
    intensity_adjusted_spectra = tf.pow(spec_array_var, 0.5)

    hparams = tf.contrib.training.HParams(
        mass_power=1.,
    )

    cos_similarity = similarity_lib.GeneralizedCosineSimilarityProvider(hparams)
    norm_spectra = cos_similarity._normalize_rows(intensity_adjusted_spectra)
    similarity = cos_similarity.compute_similarity(norm_spectra, norm_spectra)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dist = sess.run(similarity)

    return dist


def get_mainlib_replib_similarities(mainlib_spectrum, replicates_spectra):
    """Calculate the similarity between mainlib spectrum and library spectrum. 
    Args:
      mainlib_spectrum: np.array of shape ( 1000, ) 
      replicates_spectra: np.array of shape (X, 1000), where X is the number
         of replicates spectra
    Output:
      two floats, one which represents the mainlib to replicates similarity
        the other represents the overall intra-molecule similarity.
    """
    mainlib_spectrum = np.reshape(mainlib_spectrum,  (1, 1000))
    aggregate_spectra = np.concatenate((replicates_spectra, mainlib_spectrum), axis=0)

    aggregate_distance_matrix = get_similarities(aggregate_spectra)
    mainlib_average_distance = np.mean(aggregate_distance_matrix[:-1, -1 ])

    overall_average_distance = np.triu(aggregate_distance_matrix)
    num_spectra = np.shape(aggregate_distance_matrix)[0]
    num_spectra_pairs = (num_spectra - 1) * num_spectra/2
    overall_spectral_distance = (np.sum(overall_average_distance) - num_spectra) / num_spectra_pairs

    return mainlib_average_distance, overall_spectral_distance 


def get_similarity_two_spectra_sets(spectra1, spectra2):
    """Get the similarity between two spectra."""

    spectra_1 = tf.constant(spectra1, dtype=tf.float32)
    spectra_2 = tf.constant(spectra2, dtype=tf.float32)
    spectra_1 = tf.pow(spectra_1, 0.5)
    spectra_2 = tf.pow(spectra_2, 0.5)
    
    hparams = tf.contrib.training.HParams(
        mass_power=1.,
    )

    cos_similarity = similarity_lib.GeneralizedCosineSimilarityProvider(hparams)

    norm_spectra_1 = cos_similarity._normalize_rows(spectra_1)
    norm_spectra_2 = cos_similarity._normalize_rows(spectra_2)

    similarity = tf.reduce_sum(tf.multiply(norm_spectra_1, norm_spectra_2), 1, keepdims=False) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dist = sess.run(similarity)

    return dist 
    

def main():
    mol_list_17 = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/mainlib_mend.sdf')
    mainlib_inchikey_dict = train_test_split_utils.make_inchikey_dict(mol_list_17)
    replicates_mol_list_17 = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/replib_mend.sdf')
    replicates_inchikey_dict = train_test_split_utils.make_inchikey_dict(replicates_mol_list_17)

    test_ikey = 'PDACHFOTOFNHBT-UHFFFAOYSA-N' 

    mainlib_spectra = make_spectra_array(mainlib_inchikey_dict[test_ikey])
    replicates_spectra = make_spectra_array(replicates_inchikey_dict[test_ikey])

    mainlib_distance, all_spectra_distance = get_mainlib_replib_similarities(mainlib_spectra[0, : ], replicates_spectra) 

    print(mainlib_distance)
    print(all_spectra_distance)
    
if __name__ == '__main__':
    main()
