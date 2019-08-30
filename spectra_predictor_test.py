"""Tests for .spectra_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile
from absl.testing import absltest

import feature_utils
import mass_spec_constants as ms_constants
import spectra_predictor
import test_utils

import numpy as np
import tensorflow as tf


class DummySpectraPredictor(spectra_predictor.SpectraPredictor):
  """A test class that returns the mol weight input as the spectra prediction."""

  def _setup_prediction_op(self):
    fingerprint_input_op = tf.placeholder(tf.float32, (None, 4096))
    mol_weight_input_op = tf.placeholder(tf.float32, (None, 1))

    feature_dict = {
        self.fingerprint_input_key: fingerprint_input_op,
        self.molecular_weight_key: mol_weight_input_op
    }

    predict_op = tf.multiply(fingerprint_input_op, mol_weight_input_op)
    return feature_dict, predict_op


class SpectraPredictorTest(tf.test.TestCase):

  def setUp(self):
    super(SpectraPredictorTest, self).setUp()
    self.np_fingerprint_input = np.ones((2, 4096))
    self.np_mol_weight_input = np.reshape(np.array([18., 16.]), (2, 1))
    self.test_data_directory = test_utils.test_dir("testdata/")
    self.temp_dir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())
    self.test_file_short = os.path.join(self.test_data_directory,
                                        "test_2_mend.sdf")

  def tearDown(self):
    tf.reset_default_graph()
    tf.io.gfile.rmtree(self.temp_dir)
    super(SpectraPredictorTest, self).tearDown()

  def test_make_dummy_spectra_prediction(self):
    """Tests the dummy predictor."""
    predictor = DummySpectraPredictor()

    spectra_predictions = predictor.make_spectra_prediction(
        self.np_fingerprint_input, self.np_mol_weight_input)
    expected_value = np.multiply(
        self.np_fingerprint_input, self.np_mol_weight_input)
    expected_value = (
        expected_value / np.max(expected_value, axis=1, keepdims=True) *
        spectra_predictor.SCALE_FACTOR_FOR_LARGEST_INTENSITY)
    self.assertAllEqual(spectra_predictions, expected_value)

  def test_make_neims_spectra_prediction_without_weights(self):
    """Tests loading the parameters for the neims model without weights."""
    predictor = spectra_predictor.NeimsSpectraPredictor(model_checkpoint_dir="")

    spectra_predictions = predictor.make_spectra_prediction(
        self.np_fingerprint_input, self.np_mol_weight_input)

    self.assertEqual(
        np.shape(spectra_predictions),
        (np.shape(self.np_fingerprint_input)[0], ms_constants.MAX_PEAK_LOC))

  def test_load_fingerprints_from_sdf(self):
    """Tests loading fingerprints from an sdf file."""
    predictor = spectra_predictor.NeimsSpectraPredictor(model_checkpoint_dir="")

    mols_from_file = spectra_predictor.get_mol_list_from_sdf(
        self.test_file_short)
    fingerprints_from_file = predictor.get_fingerprints_from_mol_list(
        mols_from_file)

    self.assertEqual(np.shape(fingerprints_from_file), (2, 4096))

  def test_write_spectra_to_sdf(self):
    """Tests predicting and writing spectra to file."""
    predictor = spectra_predictor.NeimsSpectraPredictor(model_checkpoint_dir="")

    mols_from_file = spectra_predictor.get_mol_list_from_sdf(
        self.test_file_short)
    fingerprints, mol_weights = predictor.get_inputs_for_model_from_mol_list(
        mols_from_file)

    spectra_predictions = predictor.make_spectra_prediction(
        fingerprints, mol_weights)
    spectra_predictor.update_mols_with_spectra(mols_from_file,
                                               spectra_predictions)

    _, fpath = tempfile.mkstemp(dir=self.temp_dir)
    spectra_predictor.write_rdkit_mols_to_sdf(mols_from_file, fpath)

    # Test contents of newly written file:
    new_mol_list = spectra_predictor.get_mol_list_from_sdf(fpath)

    for idx, mol in enumerate(new_mol_list):
      peak_string_from_file = mol.GetProp(
          spectra_predictor.PREDICTED_SPECTRA_PROP_NAME)
      peak_locs, peak_intensities = feature_utils.parse_peaks(
          peak_string_from_file)
      dense_mass_spectra = feature_utils.make_dense_mass_spectra(
          peak_locs, peak_intensities, ms_constants.MAX_PEAK_LOC)
      self.assertSequenceAlmostEqual(
          dense_mass_spectra, spectra_predictions[idx, :].astype(np.int32),
          delta=1.)


if __name__ == "__main__":
  tf.test.main()
