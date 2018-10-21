import tensorflow as tf
import numpy as np

import parse_sdf_utils
import feature_map_constants as fmap_constants
import mass_spec_constants as ms_constants
import train_test_split_utils
import make_plots_from_library

record_fnames = ['/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/MAINLIB_fentanyl_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/MAINLIB_steroid_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/MAINLIB_no_family_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/REPLIB_fentanyl_TEST_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/REPLIB_steroid_TEST_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/REPLIB_fentanyl_TEST_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/REPLIB_steroid_VALIDATION_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/REPLIB_fentanyl_VALIDATION_from_mainlib.tfrecord',
                 '/mnt/storage/NIST_unzipped/fentanyl_steroid_datasets/REPLIB_steroid_VALIDATION_from_mainlib.tfrecord',
                ]
hparams = tf.contrib.training.HParams(
    max_atoms = ms_constants.MAX_ATOMS,
    max_mass_spec_peak_loc = ms_constants.MAX_PEAK_LOC,
    batch_size = 100
)

dataset = parse_sdf_utils.get_dataset_from_record(record_fnames,
        hparams,
        mode=tf.estimator.ModeKeys.TRAIN,
        features_to_load=[ fmap_constants.MOLECULE_WEIGHT])

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

max_mass = int(ms_constants.MAX_PEAK_LOC*2)
mass_bins = np.zeros((1, max_mass)) 

with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            value_dict = sess.run(next_element)
        except tf.errors.OutOfRangeError:
            break
            
        masses = value_dict[fmap_constants.MOLECULE_WEIGHT].astype(int)
        mass_bins += np.sum(np.eye(max_mass)[masses], axis=0)
        
np.save('/mnt/storage/massspec_misc/mass_bin_counts.npy', mass_bins)
