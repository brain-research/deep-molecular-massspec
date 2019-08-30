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
"""Defining a library of prediction classes for predicting mass spec data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
from os import path


import feature_map_constants as fmap_constants
import library_matching
import mass_spec_constants as ms_constants
import parse_sdf_utils
import plot_spectra_utils
import similarity as similarity_lib
import util
import numpy as np
import tensorflow as tf

MODEL_REGISTRY = {}


def register_model(model_type):
  """Registers a model by name."""

  def _decorator(model_cls):
    if model_type in MODEL_REGISTRY:
      raise ValueError('model type %s already registered' % model_type)
    MODEL_REGISTRY[model_type] = model_cls
    model_cls.model_type = type
    return model_cls

  return _decorator


class MassSpectraPrediction(object):
  """Class containing prediction and loss functions to predict mass spectra.
  """
  __metaclass__ = abc.ABCMeta

  # Will get overridden by @register_model.
  model_type = None

  @abc.abstractmethod
  def _make_learned_features(self, feature_dict, hparams, mode):
    """Make learned features given raw features."""

  @abc.abstractmethod
  def _set_model_specific_hparams(self, hparams):
    """Add additional model-specific fields to hparams."""

  @abc.abstractmethod
  def _get_model_specific_hparam_for_tuning(self):
    """Get list of model-specific hparams to tune."""

  @abc.abstractmethod
  def _get_model_specific_feature_names(self, hparams):
    """Feature keys to load from input data that are specific to the model."""

  def get_hparams_for_tuning(self, hparams):
    """Get list of names of hparams to tune."""
    hparams_to_tune = ['learning_rate']
    if hparams.loss == 'max_margin':
      hparams_to_tune.append('ranking_loss_margin')
    if hparams.bidirectional_prediction:
      hparams_to_tune.append('gate_bidirectional_predictions')

    return hparams_to_tune + self._get_model_specific_hparam_for_tuning()

  def fingerprints_to_use(self, hparams):
    if hparams.use_counting_fp:
      key = fmap_constants.COUNTING_CIRCULAR_FP_BASENAME
    else:
      key = fmap_constants.CIRCULAR_FP_BASENAME

    return str(
        ms_constants.CircularFingerprintKey(key, hparams.fp_length,
                                            hparams.radius))

  def features_to_load(self, hparams):
    """Get names of features to load."""

    feature_names = [
        library_matching.FP_NAME_FOR_JACCARD_SIMILARITY,
        fmap_constants.MOLECULE_WEIGHT, fmap_constants.DENSE_MASS_SPEC,
        fmap_constants.INCHIKEY
    ]

    if hparams.loss == 'max_margin':
      feature_names.append(fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY)

    # if hparams.reverse_prediction or hparams.bidirectional_prediction:
    feature_names.append(fmap_constants.MOLECULE_WEIGHT)

    return list(
        set(feature_names + self._get_model_specific_feature_names(hparams)))

  def get_default_hparams(self):
    """Construct default hparams values."""

    hparams = tf.contrib.training.HParams(
        init_weights='default',
        init_bias='default',
        label_names=[fmap_constants.INCHIKEY],
        max_mass_spec_peak_loc=ms_constants.MAX_PEAK_LOC,
        max_atoms=ms_constants.MAX_ATOMS,
        max_atom_type=ms_constants.MAX_ATOM_ID,
        include_atom_mass=True,
        normalize_predictions=False,
        make_spectra_plots=False,
        save_spectra_plots_to_file=False,
        do_library_matching=True,
        loss='generalized_mse',
        # When computing cosine similarity and generalized_mse, scale the
        # contribution of each spectrum coordinate i by i^mass_power.
        mass_power=1.,
        # Transform all input data, for both model training and library matching
        # evaluation by raising each peak height to this power.
        intensity_power=0.5,
        learning_rate=1e-3,
        learning_rate_decay_method='sqrt',
        learning_rate_decay_start=0,
        learning_rate_decay_scale=1000.,
        min_learning_rate_multiplier=0.05,
        gradient_clip=1.0,
        batch_size=100,
        eval_batch_size=500,
        num_inchikeys_for_plotting=5,
        ranking_loss_margin=0.05,  # Only used when loss == 'max_margin'
        resnet_bottleneck_factor=0.5,
        reverse_prediction=True,
        max_prediction_above_molecule_mass=5,
        bidirectional_prediction=True,
        gate_bidirectional_predictions=False,
        filter_library_matches_by_mass=False,
        library_matching_mass_tolerance=5,
    )

    # Subclasses add architecture-specific hparams here.
    self._set_model_specific_hparams(hparams)

    return hparams

  def _make_linear_prediction(self, learned_features, hparams):
    """A single linear layer to make the final spectra prediction."""
    return tf.layers.dense(
        inputs=learned_features,
        units=hparams.max_mass_spec_peak_loc,
        activation=None)

  def make_prediction_ops(self, feature_dict, hparams, mode, reuse=False):
    """Make prediction of molecular weight based on features.

    First, this method postprocesses the raw features predicted using
    _make_learned_features into a spectrum prediction of shape
    (, hparams.max_mass_spec_peak_loc). Then, performs additional
    postprocessing operations, such as mass_masking and relu/softmax.

    Args:
      feature_dict: Dictionary containing features parsed from TFDataset
      hparams: tf.contrib.training.HParams object. Must contain :
               max_atoms, batch_size, epochs, model_type,
               init_weight_ones, init_bias_zeros, max_mass_spec_peak_loc
      mode: whether in training or evaluation.
      reuse: whether Variables created inside this function should be reused
        from earlier calls to make_prediction.
    Returns:
      prediction: prediction of spectrum, of shape
        (batch_size, hparams.max_mass_spec_peak_loc)
      prediction_for_loss: prediction value used for downstream computation of
        the loss. The shape is the same as prediction. These may differ, for
        example, for the cross entropy loss the prediction is a probability
        vector whereas the prediction_for_loss is the corresponding logits.
    """
    with tf.variable_scope('spectrum_predictor', reuse=reuse):
      learned_features = self._make_learned_features(feature_dict, hparams,
                                                     mode)

      if hparams.bidirectional_prediction:

        forward_prediction = self._make_linear_prediction(
            learned_features, hparams)
        forward_prediction = self._mask_prediction_by_mass(
            forward_prediction, feature_dict, hparams)

        backward_prediction = self._make_linear_prediction(
            learned_features, hparams)
        backward_prediction = self._reverse_prediction(backward_prediction,
                                                       feature_dict, hparams)

        if hparams.gate_bidirectional_predictions:
          gate = tf.nn.sigmoid(
              self._make_linear_prediction(learned_features, hparams))
          raw_prediction = (
              gate * forward_prediction + (1. - gate) * backward_prediction)
        else:
          raw_prediction = (forward_prediction + backward_prediction)
      else:

        raw_prediction = self._make_linear_prediction(learned_features, hparams)

        if hparams.reverse_prediction:
          raw_prediction = self._reverse_prediction(raw_prediction,
                                                    feature_dict, hparams)
        else:
          raw_prediction = self._mask_prediction_by_mass(
              raw_prediction, feature_dict, hparams)

      if hparams.loss == 'cross_entropy':
        final_prediction = tf.nn.softmax(raw_prediction)
        prediction_for_loss = raw_prediction
      else:
        final_prediction = tf.nn.relu(raw_prediction)
        prediction_for_loss = final_prediction

      return final_prediction, prediction_for_loss

  def _make_prediction(self, feature_dict, hparams, mode, reuse=False):
    return self.make_prediction_ops(feature_dict, hparams, mode, reuse)[0]

  def make_loss(self, pred_val, feature_dict, hparams):
    """Make training loss function."""
    true_spectra = feature_dict[fmap_constants.DENSE_MASS_SPEC]
    if hparams.loss == 'generalized_mse':
      return similarity_lib.GeneralizedCosineSimilarityProvider(
          hparams).make_training_loss(true_spectra, pred_val)
    elif hparams.loss == 'cross_entropy':
      normalized_true_spectra = (
          true_spectra / tf.maximum(
              tf.reduce_sum(true_spectra, axis=1, keep_dims=True), 0.00001))
      return tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=normalized_true_spectra, logits=pred_val))
    elif hparams.loss == 'max_margin':
      target_indices = feature_dict[fmap_constants.INDEX_TO_GROUND_TRUTH_ARRAY]
      library = feature_dict['SPECTRUM_PREDICTION_LIBRARY']
      similarity_provider = similarity_lib.GeneralizedCosineSimilarityProvider(
          hparams)
      return similarity_lib.max_margin_ranking_loss(
          pred_val, target_indices, library, similarity_provider,
          hparams.ranking_loss_margin)
    else:
      raise ValueError('loss type %s not supported' % hparams.loss)

  def _mask_prediction_by_mass(self, raw_prediction, feature_dict, hparams):
    """Zero out predictions to the right of the maximum possible mass."""

    total_mass = feature_dict[fmap_constants.MOLECULE_WEIGHT][..., 0]
    total_mass = tf.cast(tf.round(total_mass), dtype=tf.int32)

    # We mask out things that are to the right of total mass
    indices = np.arange(raw_prediction.shape[-1].value)[np.newaxis, ...]
    right_of_total_mass = indices > (
        total_mass[..., tf.newaxis] +
        hparams.max_prediction_above_molecule_mass)

    return tf.where(right_of_total_mass, tf.zeros_like(raw_prediction),
                    raw_prediction)

  def _reverse_prediction(self, raw_prediction, feature_dict, hparams):
    total_mass = feature_dict[fmap_constants.MOLECULE_WEIGHT][..., 0]
    total_mass = tf.cast(tf.round(total_mass), dtype=tf.int32)
    return util.scatter_by_anchor_indices(
        total_mass, raw_prediction, hparams.max_prediction_above_molecule_mass)

  def make_evaluation_metrics(self, feature_dict, hparams, dataset_config_file,
                              output_dir):
    """Make a dict of Estimator-compatible evaluation metrics."""

    make_prediction = functools.partial(
        self._make_prediction,
        hparams=hparams,
        mode=tf.estimator.ModeKeys.EVAL,
        reuse=tf.AUTO_REUSE)
    if hparams.do_library_matching:
      # The library matching is generally very memory-intensive, so we
      # place all of its computation on the CPU.
      with tf.device('/CPU:0'):
        prediction_log_dir = path.join(output_dir, 'predictions')
        tf.gfile.MakeDirs(prediction_log_dir)
        mass_tolerance = (
            hparams.library_matching_mass_tolerance
            if hparams.filter_library_matches_by_mass else None)
        (metrics, library_match_spectra,
         library_match_inchikeys) = library_matching.library_match_accuracy(
             feature_dict[fmap_constants.LIBRARY_MATCHING],
             make_prediction,
             hparams.eval_batch_size,
             similarity_provider=similarity_lib.
             GeneralizedCosineSimilarityProvider(hparams),
             mass_tolerance=mass_tolerance,
             log_dir=prediction_log_dir)
    else:
      metrics = {}

    if hparams.make_spectra_plots:
      data_for_spectra_plots = feature_dict[fmap_constants.SPECTRUM_PREDICTION]
      pred_val = self._make_prediction(
          data_for_spectra_plots,
          hparams,
          tf.estimator.ModeKeys.EVAL,
          reuse=True)

      inchikey_list_for_plotting = plot_spectra_utils.inchikeys_for_plotting(
          dataset_config_file, hparams.num_inchikeys_for_plotting,
          hparams.eval_batch_size)

      # Undo the effect of any preprocessing done to the input spectra.
      inchikey_list = data_for_spectra_plots[fmap_constants.INCHIKEY]
      true_spectra = data_for_spectra_plots[fmap_constants.DENSE_MASS_SPEC]

      pred_val = parse_sdf_utils.postprocess_spectrum(pred_val, hparams)
      true_spectra = parse_sdf_utils.postprocess_spectrum(true_spectra, hparams)

      if hparams.do_library_matching:
        library_match_spectra = parse_sdf_utils.postprocess_spectrum(
            library_match_spectra, hparams)

      for inchikey in inchikey_list_for_plotting:
        inchikey = inchikey.strip()
        if hparams.save_spectra_plots_to_file:
          spectra_plot_dir = output_dir
        else:
          spectra_plot_dir = ''

        predict_spectra_keyname = plot_spectra_utils.name_metric(
            plot_spectra_utils.PlotModeKeys.PREDICTED_SPECTRUM, inchikey)
        metrics[predict_spectra_keyname] = (
            plot_spectra_utils.spectra_plot_summary_op(
                inchikey_list,
                true_spectra,
                pred_val,
                inchikey,
                plot_mode_key=plot_spectra_utils.PlotModeKeys.
                PREDICTED_SPECTRUM,
                image_directory=spectra_plot_dir))

        if hparams.do_library_matching:
          lib_match_keyname = plot_spectra_utils.name_metric(
              plot_spectra_utils.PlotModeKeys.LIBRARY_MATCHED_SPECTRUM,
              inchikey)
          metrics[lib_match_keyname] = (
              plot_spectra_utils.spectra_plot_summary_op(
                  inchikey_list,
                  true_spectra,
                  library_match_spectra,
                  inchikey,
                  plot_mode_key=plot_spectra_utils.PlotModeKeys.
                  LIBRARY_MATCHED_SPECTRUM,
                  library_match_inchikeys=library_match_inchikeys,
                  image_directory=spectra_plot_dir))

    return metrics


@register_model('mlp')
class MLPSpectraPrediction(MassSpectraPrediction):
  """Prediction with a multi-layer perceptron."""

  def _set_model_specific_hparams(self, hparams):
    hparams.add_hparam('num_hidden_units', 2000)
    hparams.add_hparam('hidden_layer_activation', 'relu')
    hparams.add_hparam('dropout_rate', 0.25)
    hparams.add_hparam('num_hidden_layers', 1)
    hparams.add_hparam('use_counting_fp', True)
    hparams.add_hparam('fp_length', 4096)
    hparams.add_hparam('radius', 2)

  def _get_model_specific_feature_names(self, hparams):
    return [self.fingerprints_to_use(hparams)]

  def _get_model_specific_hparam_for_tuning(self):
    return ['num_hidden_units', 'dropout_rate', 'num_hidden_layers']

  def _batch_norm(self, features, is_training):
    return tf.layers.batch_normalization(features, training=is_training)

  def _residual_block(self, features, hparams, activation_fn, is_training):
    """Construct a single block for a residual network."""

    features = self._batch_norm(features, is_training)
    features = activation_fn(features)
    features = tf.layers.dropout(
        inputs=features, rate=hparams.dropout_rate, training=is_training)
    features = tf.layers.dense(
        features, (hparams.resnet_bottleneck_factor * hparams.num_hidden_units))
    features = self._batch_norm(features, is_training=is_training)
    features = activation_fn(features)
    return tf.layers.dense(features, hparams.num_hidden_units)

  def _make_learned_features(self, feature_dict, hparams, mode):

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    activation_fn = getattr(tf.nn, hparams.hidden_layer_activation)

    feature_to_use = self.fingerprints_to_use(hparams)
    layer_output = feature_dict[feature_to_use]

    if hparams.num_hidden_layers > 0:
      layer_output = tf.layers.dense(
          inputs=layer_output,
          units=hparams.num_hidden_units,
          activation=activation_fn)

    for _ in range(hparams.num_hidden_layers):
      layer_output += self._residual_block(layer_output, hparams, activation_fn,
                                           is_training)

    layer_output = self._batch_norm(layer_output, is_training)
    return activation_fn(layer_output)


@register_model('linear')
class LinearSpectraPrediction(MLPSpectraPrediction):

  def _set_model_specific_hparams(self, hparams):
    super(LinearSpectraPrediction, self)._set_model_specific_hparams(hparams)
    hparams.set_hparam('num_hidden_layers', 0)

  def _get_model_specific_hparam_for_tuning(self):
    return ['dropout_rate']


@register_model('smiles_rnn')
class SmilesRNNSpectraPrediction(MassSpectraPrediction):
  """RNN applied to SMILES representation."""

  def _set_model_specific_hparams(self, hparams):
    hparams.add_hparam('num_rnn_hidden_units', 500)
    hparams.add_hparam('embedding_dim', 10)
    hparams.add_hparam('average_rnn_outputs', True)

  def _get_model_specific_feature_names(self, hparams):
    return [fmap_constants.SMILES]

  def _get_model_specific_hparam_for_tuning(self):
    return ['num_rnn_hidden_units', 'embedding_dim', 'average_rnn_outputs']

  def _make_learned_features(self, feature_dict, hparams, mode):
    sequence_length = feature_dict[fmap_constants.SMILES_TOKEN_LIST_LENGTH]
    embedding_table = tf.get_variable(
        'atom_embeddings',
        [len(ms_constants.SMILES_TOKEN_NAMES), hparams.embedding_dim])
    processed_features = tf.nn.embedding_lookup(
        embedding_table, feature_dict[fmap_constants.SMILES])

    fw_rnn_cell = tf.nn.rnn_cell.LSTMCell(hparams.num_rnn_hidden_units)
    bw_rnn_cell = tf.nn.rnn_cell.LSTMCell(hparams.num_rnn_hidden_units)
    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        fw_rnn_cell,
        bw_rnn_cell,
        processed_features,
        sequence_length=sequence_length,
        dtype=tf.float32)

    # Concatenate forward and backward outputs
    rnn_outputs = tf.concat(rnn_outputs, 2)

    if hparams.average_rnn_outputs:
      rnn_outputs = (
          tf.reduce_sum(rnn_outputs, axis=1) / tf.cast(
              sequence_length[..., tf.newaxis], tf.float32))
    else:
      rnn_outputs = rnn_outputs[:, -1, ...]  # Take the final output.

    return rnn_outputs


@register_model('baseline')
class BaselinePrediction(MLPSpectraPrediction):
  """Tune hparams for tuning to mass_power and intensity_power for baseline."""

  def get_hparams_for_tuning(self, hparams):
    del hparams
    return ['mass_power', 'intensity_power']

  def _set_model_specific_hparams(self, hparams):
    super(BaselinePrediction, self)._set_model_specific_hparams(hparams)
    hparams.set_hparam('use_counting_fp', False)
    hparams.set_hparam('num_hidden_layers', 0)
    hparams.set_hparam('bidirectional_prediction', False)
    hparams.set_hparam('reverse_prediction', False)


def get_prediction_helper(model_type):
  """Provide the correct prediction helper, based on model type."""
  if model_type not in MODEL_REGISTRY:
    raise ValueError('Unrecognized model type: %s' % model_type)
  return MODEL_REGISTRY[model_type]()
