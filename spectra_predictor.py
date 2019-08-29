"""Helpers for generating spectra prediction from trained models."""

import abc

import feature_map_constants as fmap_constants
import feature_utils
import mass_spec_constants as ms_constants
import molecule_predictors

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import six
import tensorflow as tf

_DEFAULT_HPARAMS = {
    "radius": 2,
    "mass_power": 1.0,
    "gate_bidirectional_predictions": True,
    "include_atom_mass": True,
    "init_bias": "default",
    "reverse_prediction": True,
    "max_mass_spec_peak_loc": 1000,
    "num_hidden_units": 2000,
    "use_counting_fp": True,
    "max_atoms": 100,
    "intensity_power": 0.5,
    "max_prediction_above_molecule_mass": 5,
    "fp_length": 4096,
    "bidirectional_prediction": True,
    "resnet_bottleneck_factor": 0.5,
    "max_atom_type": 100,
    "hidden_layer_activation": "relu",
    "init_weights": "default",
    "num_hidden_layers": 7
}
_DEFAULT_HPARAMS_STR = ",".join(
    "{}={}".format(k, v) for k, v in six.iteritems(_DEFAULT_HPARAMS))
PREDICTED_SPECTRA_PROP_NAME = "PREDICTED SPECTRUM"

# Predictions from the model are normalized by default.
# This factor is used to rescale the predictions so the highest intensity has
# this value.
SCALE_FACTOR_FOR_LARGEST_INTENSITY = 999.


def fingerprints_to_use(hparams):
  """Given tf.HParams, return a ms_constants.CircularFingerprintKey."""
  if hparams.use_counting_fp:
    key = fmap_constants.COUNTING_CIRCULAR_FP_BASENAME
  else:
    key = fmap_constants.CIRCULAR_FP_BASENAME

  return ms_constants.CircularFingerprintKey(key, hparams.fp_length,
                                             hparams.radius)


def get_mol_weights_from_mol_list(mol_list):
  """Given a list of rdkit.Mols, return weights for each mol."""
  return np.array([Chem.rdMolDescriptors.CalcExactMolWt(m) for m in mol_list])


def get_mol_list_from_sdf(sdf_fname):
  """Reads a sdf file and returns a list of molecules.

  Note: rdkit's Chem.SDMolSupplier only accepts filenames as inputs. As such
  this code only supports local filesystem name environments.

  Args:
    sdf_fname: Path to sdf file.

  Returns:
    List of rdkit.Mol objects.

  Raises:
    ValueError if a molblock in the SDF cannot be parsed.
  """
  suppl = Chem.SDMolSupplier(sdf_fname)
  mols = []

  for idx, mol in enumerate(suppl):
    if mol is not None:
      mols.append(mol)
    else:
      fail_sdf_block = suppl.GetItemText(idx)
      raise ValueError("Unable to parse the following mol block %s" %
                       fail_sdf_block)
  return mols


def update_mols_with_spectra(mol_list, spectra_array):
  """Writes a predicted spectrum for each RDKit.mol object.

  Args:
    mol_list: List of rdkit.Mol objects.
    spectra_array: np.array of spectra.

  Returns:
    Updated list of rdkit.Mol objects where each molecule contains a predicted
      spectrum.
  """
  if len(mol_list) != np.shape(spectra_array)[0]:
    raise ValueError("Number of mols in mol list %d is not equal to number of "
                     "spectra found %d." %
                     (len(mol_list), np.shape(spectra_array)[0]))
  for mol, spectrum in zip(mol_list, spectra_array):
    spec_array_text = feature_utils.convert_spectrum_array_to_string(spectrum)
    mol.SetProp(PREDICTED_SPECTRA_PROP_NAME, spec_array_text)
  return mol_list


def write_rdkit_mols_to_sdf(mol_list, out_sdf_name):
  """Writes a series of rdkit.Mol to SDF.

  Args:
    mol_list: List of rdkit.Mol objects.
    out_sdf_name: Output file path for molecules.
  """
  writer = AllChem.SDWriter(out_sdf_name)

  for mol in mol_list:
    writer.write(mol)
  writer.close()


class SpectraPredictor(object):
  """Helper for generating a computational graph for making predictions."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, hparams_str=""):
    """Sets up graph, session, and input and output ops for prediction.

    Args:
      hparams_str (str): String containing hyperparameter settings.
    """

    self._prediction_helper = molecule_predictors.get_prediction_helper("mlp")
    self._hparams = self._prediction_helper.get_default_hparams()
    self._hparams.parse(hparams_str)
    self._fingerprint_key = fingerprints_to_use(self._hparams)
    self.fingerprint_input_key = str(self._fingerprint_key)
    self.molecular_weight_key = fmap_constants.MOLECULE_WEIGHT

    self._graph = tf.Graph()
    self._sess = tf.Session(graph=self._graph)
    with self._graph.as_default():
      (self._placeholder_dict, self._predict_op) = self._setup_prediction_op()
    assert set(self._placeholder_dict) == set(
        [self.fingerprint_input_key, self.molecular_weight_key])

  @abc.abstractmethod
  def _setup_prediction_op(self):
    """Sets up prediction operation.

    Returns:
      placeholder_dict: Dict with self.fingerprint_input_key and
        self.molecular_weight_key as keys and values which are tf.placeholder
        for predicted spectra.
      predict_op: tf.Tensor for predicted spectra.
    """

  def make_spectra_prediction(self, fingerprint_array, molecule_weight_array):
    """Makes spectra prediction.

    Args:
      fingerprint_array (np.array): Contains molcule fingerprints.
      molecule_weight_array (np.array): Contains molecular weights. Should have
        same batch dimension as fingerprint_array.

    Returns:
      np.array of predictions.
    """
    molecule_weight_array = np.reshape(molecule_weight_array, (-1, 1))
    with self._graph.as_default():
      prediction = self._sess.run(
          self._predict_op,
          feed_dict={
              self._placeholder_dict[self.fingerprint_input_key]:
                  fingerprint_array,
              self._placeholder_dict[self.molecular_weight_key]:
                  molecule_weight_array
          })

    prediction = prediction / np.max(
        prediction, axis=1, keepdims=True) * SCALE_FACTOR_FOR_LARGEST_INTENSITY
    return prediction

  def get_fingerprints_from_mol_list(self, mol_list):
    """Converts a list of rdkit.Mol objects into circular fingerprints.

    Args:
      mol_list: List of rdkit.Mol objects.

    Returns:
      np.array of fingerprints for prediction.
    """

    fingerprints = [
        feature_utils.make_circular_fingerprint(mol, self._fingerprint_key)
        for mol in mol_list
    ]

    return np.array(fingerprints)

  def get_inputs_for_model_from_mol_list(self, mol_list):
    """Grabs fingerprints and molecular weights for the prediction model."""
    fingerprints = self.get_fingerprints_from_mol_list(mol_list)
    weights = get_mol_weights_from_mol_list(mol_list)
    return fingerprints, weights


class NeimsSpectraPredictor(SpectraPredictor):
  """Helper for making spectra predictions using the trained NEIMS model."""

  def __init__(self, model_checkpoint_dir, hparams_str=_DEFAULT_HPARAMS_STR):
    """Initializes the predictor with the weights and hyperparameters.

    Args:
      model_checkpoint_dir (str): Path to checkpoint weights.
      hparams_str (str): String that contains hyperparameters for model.
    """
    super(NeimsSpectraPredictor, self).__init__(hparams_str)
    self.restore_from_checkpoint(model_checkpoint_dir)

  def _setup_prediction_op(self):
    """Sets up prediction operation and inputs for model."""
    fp_length = self._hparams.fp_length

    fingerprint_input_op = tf.placeholder(tf.float32, (None, fp_length))
    mol_weight_input_op = tf.placeholder(tf.float32, (None, 1))

    feature_dict = {
        self.fingerprint_input_key: fingerprint_input_op,
        self.molecular_weight_key: mol_weight_input_op
    }

    predict_op, _ = self._prediction_helper.make_prediction_ops(
        feature_dict,
        self._hparams,
        mode=tf.estimator.ModeKeys.PREDICT,
        reuse=False)

    return feature_dict, predict_op

  def restore_from_checkpoint(self, model_checkpoint_dir):
    """Restores model parameters from checkpoint directory.

    Args:
      model_checkpoint_dir (str): filepath directory to weights. If empty, model
        will be initialized with random weights.
    """
    with self._graph.as_default():
      if model_checkpoint_dir:
        saver = tf.train.Saver()
        saver.restore(self._sess,
                      tf.train.latest_checkpoint(model_checkpoint_dir))
      else:
        tf.logging.warn("No model checkpoint directory given,"
                        " reinitializing model.")
        self._sess.run(tf.global_variables_initializer())
