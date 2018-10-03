from rdkit import Chem
import numpy as np
import tensorflow as tf

import parse_sdf_utils
import train_test_split_utils
import feature_utils
import mass_spec_constants as ms_constants
import molecule_predictors
import similarity as similarity_lib


def make_spectra_array(inchikey, inchikey_dict):
    """Grab spectra pertaining to same molecule in one np.array.
    Args:
      inchikey: inchikey (should be present in the inchikey dict)
      inchikey_dict: A dict keyed on inchikey, containing lists of rdkit.Mol objects
         Each Mol should contain information about the spectra, as stored in NIST.
    Output: 
      np.array of spectra of shape (number of spectra, max spectra length)
    """
    mass_spec_spectra = np.zeros( ( len(inchikey_dict[inchikey]), ms_constants.MAX_PEAK_LOC))
    for idx, mol in enumerate(inchikey_dict[inchikey]):
        spectra_str = mol.GetProp(ms_constants.SDF_TAG_MASS_SPEC_PEAKS)
        spectral_locs, spectral_intensities = feature_utils.parse_peaks(spectra_str)
        dense_mass_spec = feature_utils.make_dense_mass_spectra(
            spectral_locs, spectral_intensities, ms_constants.MAX_PEAK_LOC)
    
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
    sqrt_spectra = tf.sqrt(spec_array_var)

    prediction_helper = molecule_predictors.get_prediction_helper('mlp')
    hparams = prediction_helper.get_default_hparams()

    cos_similarity = similarity_lib.GeneralizedCosineSimilarityProvider(hparams)
    norm_spectra = cos_similarity._normalize_rows(sqrt_spectra)
    distance = cos_similarity.compute_similarity(norm_spectra, norm_spectra)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dist = sess.run(distance)

    return dist


def main():
    mol_list = parse_sdf_utils.get_sdf_to_mol('/mnt/storage/NIST_zipped/NIST17/replib_mend.sdf')
    inchikey_dict = train_test_split_utils.make_inchikey_dict(mol_list)

    spectra_for_one_mol = make_spectra_array('PDACHFOTOFNHBT-UHFFFAOYSA-N', inchikey_dict)
    distance_matrix = get_similarities(spectra_for_one_mol)
    print('distance for spectra in PDACHFOTOFNHBT-UHFFFAOYSA-N', distance_matrix)
    
if __name__ == '__main__':
    main()
