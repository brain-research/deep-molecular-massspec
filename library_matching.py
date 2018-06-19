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
"""Evaluation metric for accuracy of library matching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
from os import path

import feature_map_constants as fmap_constants
import mass_spec_constants as ms_constants
import util
import numpy as np
import tensorflow as tf

FP_NAME_FOR_JACCARD_SIMILARITY = str(
    ms_constants.CircularFingerprintKey(fmap_constants.CIRCULAR_FP_BASENAME,
                                        1024, 2))

# When filtering the library matching candidates on a per-query basis, we
# set the query-library element similarity to this value for the elements
# that were filtered.
_SIMILARITY_FOR_FILTERED_ELEMENTS = -100000

_KEY_FOR_LIBRARY_VECTORS = fmap_constants.DENSE_MASS_SPEC


def _validate_data_dict(data_dict, name):
  if data_dict is None:
    return
  for key in [
      FP_NAME_FOR_JACCARD_SIMILARITY, fmap_constants.INCHIKEY,
      _KEY_FOR_LIBRARY_VECTORS, fmap_constants.MOLECULE_WEIGHT
  ]:
    if key not in data_dict:
      raise ValueError('input dataset with name %s '
                       'must have field %s' % (name, key))


class LibraryMatchingData(
    namedtuple('LibraryMatchingData', ['observed', 'predicted', 'query'])):
  """Data for library matching evaluation.

  All input data dictionaries must have the following keys:
  fmap_constants.INCHIKEY, fmap_constants.DENSE_MASS_SPEC,
  and all keys in fmap_constants.FINGERPRINT_LIST

  Args:
    observed: data put into the library using true observed spectra.
    predicted: data put into the library using the output of a predictive model
      applied to the data's molecules.
    query: data containing observed spectra used to issue queries to the
      library.
  """

  def __new__(cls, observed, predicted, query):
    _validate_data_dict(observed, 'observed')
    _validate_data_dict(predicted, 'predicted')
    _validate_data_dict(query, 'query')

    return super(LibraryMatchingData, cls).__new__(cls, observed, predicted,
                                                   query)


def _invert_permutation(perm):
  """Convert an array of permutations to an array of inverse permutations.

  Args:
    perm: a [batch_size, num_iterms] int array where each column is a
      permutation.
  Returns:
    A [batch_size, num_iterms] int array where each column is the
    inverse permutation of the corresponding input column.
  """

  output = np.empty(shape=perm.shape, dtype=perm.dtype)
  output[np.arange(perm.shape[0])[..., np.newaxis], perm] = np.arange(
      perm.shape[1])[np.newaxis, ...]
  return output


def _find_query_rank_helper(similarities, library_keys, query_keys):
  """Find rank of query key when we sort library_keys by similarities.

  Note that the behavior of this function is not well defined when there
  are ties along a row of similarities.

  Args:
    similarities: [batch_size, num_library_elements] float array. These are not
      assumed to be sorted in any way.
    library_keys: [num_library_elements] string array, where each column j of
      similarities corresponds to library_key j.
    query_keys: [num_queries] string array.

  Returns:
    highest_rank: A [batch_size] tf.int32  np array containing
      for each batch the highest index of a library key that matches the query
      key for that batch element when the library keys are sorted in descending
      order by similarity score.
    lowest_rank: similar to highest_rank, but the lowest index of a library key
      matchign the query.
    avg_rank: A [batch_size] tf.float32 array containing the average index
      of all library keys matching the query.
    best_query_similarities: the value of the similarities evaluated at
      the lowest_rank position.
  Raises:
    ValueError: if there is a query key does not exist in the set of library
      keys.
  """

  def _masked_rowwise_max(data, mask):
    data = np.copy(data)
    min_value = np.min(data) - 1
    data[~mask] = min_value
    return np.max(data, axis=1)

  def _masked_rowwise_min(data, mask):
    data = np.copy(data)
    max_value = np.max(data) + 1
    data[~mask] = max_value
    return np.min(data, axis=1)

  def _masked_rowwise_mean(data, mask):
    masked_data = data * mask
    rowwise_value_sum = np.float32(np.sum(masked_data, axis=1))
    rowwise_count = np.sum(mask, axis=1)
    return rowwise_value_sum / rowwise_count

  library_key_matches_query = (
      library_keys[np.newaxis, ...] == query_keys[..., np.newaxis])
  if not np.all(np.any(library_key_matches_query, axis=1)):
    raise ValueError('Not all query keys appear in the library.')

  ranks = _invert_permutation(np.argsort(-similarities, axis=1))

  highest_rank = _masked_rowwise_max(ranks, library_key_matches_query)
  lowest_rank = _masked_rowwise_min(ranks, library_key_matches_query)
  avg_rank = _masked_rowwise_mean(ranks, library_key_matches_query)

  highest_rank = np.int32(highest_rank)
  lowest_rank = np.int32(lowest_rank)
  avg_rank = np.float32(avg_rank)

  best_query_similarities = _masked_rowwise_max(similarities,
                                                library_key_matches_query)

  return (highest_rank, lowest_rank, avg_rank, best_query_similarities)


def _find_query_rank(similarities, library_keys, query_keys):
  """tf.py_func wrapper around _find_query_rank_helper.

  Args:
    similarities: [batch_size, num_library_elements] float Tensor. These are not
      assumed to be sorted in any way.
    library_keys: [num_library_elements] string Tensor, where each column j of
      similarities corresponds to library_key j.
    query_keys: [num_queries] string Tensor
  Returns:
    query_ranks: a dictionary with keys 'highest', 'lowest' and 'avg', where
      each value is a [batch_size] Tensor. The 'lowest' Tensor contains
      for each batch the lowest index of a library key that matches the query
      key for that batch element when the library keys are sorted in descending
      order by similarity score. The 'highest' and 'avg'
      Tensors are defined similarly. The first two are tf.int32 and the
      final is a tf.float32.

      Note that the behavior of these metrics is undefined when there are ties
      within a row of similarities.
    best_query_similarities: the value of the similarities evaluated at
      the lowest query rank.
  """

  (highest_rank, lowest_rank, avg_rank, best_query_similarities) = tf.py_func(
      _find_query_rank_helper, [similarities, library_keys, query_keys],
      (tf.int32, tf.int32, tf.float32, tf.float32),
      stateful=False)

  query_ranks = {
      'highest': highest_rank,
      'lowest': lowest_rank,
      'avg': avg_rank
  }

  return query_ranks, best_query_similarities


def _max_similarity_match(library,
                          query,
                          similarity_provider,
                          library_filter=None,
                          library_keys=None,
                          query_keys=None):
  """Find maximum similarity between query and library vectors.

  All queries and library elements are associated with a key. We require that
  each query key exists in the library. In other words, the ground truth id
  of the query is in the library. We optionally return additional data
  about how similar the library element for the ground truth was to the query.

  Args:
    library: [num_elements, feature_dim] Tensor
    query: [num_queries, feature_dim] Tensor
    similarity_provider: a similarity.SimilarityProvider instance
    library_filter: [num_elements, num_queries] Bool Tensor. Query j is
      permitted to be matched to library element i if library_filter[i, j] is
      True. Unused if None.
    library_keys: tf.string Tensor of the ids of the library elements
    query_keys: tf.string Tensor of the ids of the queries

  Returns:
    argmax: [num_queries] tf.int32 Tensor containing indices of maximum inner
      product between each query and the library.
    best_similarities: [num_queries] tf.float32 Tensor containing the value of
      the maximum inner products.
    query_ranks: a dictionary with keys 'highest', 'lowest' and 'avg', where
      each value is a [batch_size] Tensor. The 'lowest' Tensor contains
      for each batch the lowest index of a library key that matches the query
      key for that batch element when the library keys are sorted in descending
      order by similarity score. The 'highest' and 'avg'
      Tensors are defined similarly. The first two are tf.int32 and the
      final is a tf.float32.

      Note that the behavior of these metrics is undefined when there are ties
      within a row of similarities.
    query_similarities: [num_queries] corresponding similarities for the
      'lowest' ranks in query_ranks above.
    library_entry_of_predictions: [num_queries, feature_dim] Tensor
  """

  similarities = similarity_provider.compute_similarity(library, query)
  if library_filter is not None:
    error_tensors = [
        'For some query, all elements of the library were '
        'removed by filtering.'
    ]
    if query_keys is not None:
      error_tensors.append(query_keys)
    assert_op = tf.Assert(
        tf.reduce_all(tf.reduce_any(library_filter, axis=0)), error_tensors)
    with tf.control_dependencies([assert_op]):
      library_filter = tf.transpose(library_filter, (1, 0))
    similarities = tf.where(
        library_filter, similarities,
        _SIMILARITY_FOR_FILTERED_ELEMENTS * tf.ones_like(similarities))
  argmax = tf.argmax(similarities, axis=1)
  row_indices = tf.range(0, tf.shape(argmax)[0])
  argmax_with_indices = tf.stack(
      [row_indices, tf.cast(argmax, tf.int32)], axis=1)
  best_similarities = tf.gather_nd(similarities, argmax_with_indices)
  library_entry_of_prediction = tf.gather(library, argmax)
  library_entry_of_prediction = similarity_provider.undo_library_preprocessing(
      library_entry_of_prediction)

  if library_keys is not None and query_keys is not None:
    if library_keys.shape.ndims == 2:
      library_keys = tf.squeeze(library_keys, axis=1)
    if query_keys.shape.ndims == 2:
      query_keys = tf.squeeze(query_keys, axis=1)

    query_ranks, query_similarities = _find_query_rank(similarities,
                                                       library_keys, query_keys)
  else:
    query_similarities = None
    query_ranks = None

  return (argmax, best_similarities, query_ranks, query_similarities,
          library_entry_of_prediction)


def _make_library(predicted_dict,
                  predictor_fn,
                  observed_dict,
                  eval_batch_size,
                  similarity_provider,
                  name='library'):
  """Make idempotent [num_elements, library_entry_length] library Tensor."""

  def _get_library_shape(predicted_dict, observed_library):
    """Infer the shape of the library from the observed and predicted data."""

    if observed_library is None:
      prediction_shape = util.get_static_shape_without_adding_ops(
          predicted_dict, predictor_fn)
      library_entry_length = prediction_shape[1]
      num_elements_observed = 0
    else:
      (num_elements_observed,
       library_entry_length) = observed_library.shape.as_list()
      # Having a statically-inferrable batch size is required, since we need to
      # know the exact shape of the constructed library at graph construction
      # time, since it will be stored in a tf.Variable.
      assert num_elements_observed is not None, ('batch_size must be '
                                                 'statically inferrable for '
                                                 'the observed data.')

    num_elements = num_elements_observed
    if predicted_dict is not None:
      num_elements_predicted = tf.contrib.framework.nest.flatten(
          predicted_dict)[0].shape[0]
      assert num_elements_predicted is not None, ('batch_size must be '
                                                  'statically inferrable for '
                                                  'the predicted data.')
      num_elements += num_elements_predicted
    return [num_elements, library_entry_length]

  if observed_dict is not None:
    observed_library = observed_dict[_KEY_FOR_LIBRARY_VECTORS]
  else:
    observed_library = None

  if predicted_dict is not None:
    library_shape = _get_library_shape(predicted_dict, observed_library)

    # The library may require expensive computation to construct. Therefore
    # at evaluation time we do this computation once and cache the result in a
    # Variable. The first function below allocates this Variable. The second
    # creates the potentially-expensive operation for setting the Variable to
    # the desired value.
    def make_value_op():
      # It's important to use a local variable rather than a global Variable.
      # Global variables get restored from checkpoints. This would be bad here,
      # since we want to recompute the library with respect to the predictions
      # of the current model.
      return tf.get_local_variable(
          name=name,
          shape=library_shape,
          dtype=tf.float32,
          initializer=tf.zeros_initializer)

    def make_init_op(value_op):
      prediction = util.map_predictor(
          predicted_dict, predictor_fn, sub_batch_size=eval_batch_size)
      if observed_dict is not None:
        library = tf.concat([prediction, observed_library], axis=0)
      else:
        library = prediction
      normalized_library = similarity_provider.preprocess_library(library)
      return value_op.assign(normalized_library)

    full_library = util.value_op_with_initializer(make_value_op, make_init_op)
  else:
    full_library = similarity_provider.preprocess_library(observed_library)

  def _get_ids_fingerprints_and_masses(data_dict):
    if data_dict is None:
      return [], [], []
    ids = data_dict[fmap_constants.INCHIKEY]
    if ids.shape[0] == 0:
      return [], [], []
    fingerprints = data_dict[FP_NAME_FOR_JACCARD_SIMILARITY]
    masses = tf.squeeze(data_dict[fmap_constants.MOLECULE_WEIGHT], axis=1)
    return [ids], [fingerprints], [masses]

  (predicted_ids, predicted_fingerprints,
   predicted_masses) = _get_ids_fingerprints_and_masses(predicted_dict)
  (observed_ids, observed_fingerprints,
   observed_masses) = _get_ids_fingerprints_and_masses(observed_dict)

  full_library_ids = tf.concat(predicted_ids + observed_ids, axis=0)

  full_fingerprints = tf.concat(
      predicted_fingerprints + observed_fingerprints, axis=0)

  full_masses = tf.concat(predicted_masses + observed_masses, axis=0)
  return full_library, full_library_ids, full_fingerprints, full_masses


def library_matching(combined_data,
                     predictor_fn,
                     similarity_provider,
                     mass_tolerance,
                     eval_batch_size=500):
  """Classify query spectra using a library of observed and predicted spectra.

  We first construct a library of spectra by merging a set of observed spectra
  with a set of spectra that are generated synthetically using a predictive
  model. Each spectrum in the library is associated with a the id of the true
  molecule that it is associated with.

  Next, we stream over a set of query spectra and compute the cosine similarity
  between the each query and each element of the library. For each query, we
  output the id of the library spectrum that it is most similar to.

  Args:
    combined_data: a LibraryMatchingData instance
    predictor_fn: a callable that takes such a data dict and returns a predicted
      spectrum.
    similarity_provider: A similarity.SimilarityProvider instance.
    mass_tolerance: library elements are only considered as candidate
      matches if their mass is within this much of the query mass. If None,
      no filtering is performed.
    eval_batch_size: int for the batch size to use when predicting spectra to
      include in the library.
  Returns:
    true_ids: string Tensor containing the ground truth ids for the queries.
    predicted_ids: string Tensor contain the ids of the elements of the library
      that the queries were matched to.
    library_entry_of_prediction: float Tensor containing the library spectra
      that is the best match for the query
    num_library_elements: int
  """
  observed_dict = combined_data.observed
  predicted_dict = combined_data.predicted
  query_dict = combined_data.query

  (full_library, full_library_ids, full_fingerprints,
   full_masses) = _make_library(predicted_dict, predictor_fn, observed_dict,
                                eval_batch_size, similarity_provider)

  true_ids = query_dict[fmap_constants.INCHIKEY]
  query = similarity_provider.preprocess_queries(
      query_dict[fmap_constants.DENSE_MASS_SPEC])

  if mass_tolerance is not None:
    query_masses = tf.squeeze(
        query_dict[fmap_constants.MOLECULE_WEIGHT], axis=1)[tf.newaxis, ...]
    full_masses = full_masses[..., tf.newaxis]
    library_filter = tf.abs(query_masses - full_masses) <= mass_tolerance
  else:
    library_filter = None

  (library_match_indices, best_similarities, query_ranks,
   query_similarities, library_entry_of_prediction) = _max_similarity_match(
       full_library, query, similarity_provider, library_filter,
       full_library_ids, true_ids)

  predicted_ids = tf.gather(full_library_ids, library_match_indices)

  true_fingerprints = query_dict[FP_NAME_FOR_JACCARD_SIMILARITY]
  predicted_fingerprints = tf.gather(full_fingerprints, library_match_indices)

  true_data = {
      fmap_constants.INCHIKEY: true_ids,
      FP_NAME_FOR_JACCARD_SIMILARITY: true_fingerprints,
      'similarity': query_similarities,
      'rank': query_ranks
  }
  predicted_data = {
      fmap_constants.INCHIKEY: predicted_ids,
      FP_NAME_FOR_JACCARD_SIMILARITY: predicted_fingerprints,
      'similarity': best_similarities,
  }
  num_library_elements = full_library_ids.shape[0].value

  return (true_data, predicted_data, library_entry_of_prediction,
          num_library_elements)


def _log_predictions(true_keys, predicted_keys, ranks, global_step, log_dir):
  output_file = path.join(log_dir,
                          '%d.library_matching_predictions.txt' % global_step)
  with tf.gfile.Open(output_file, 'w') as f:
    for true_key, predicted_key, rank in zip(true_keys, predicted_keys, ranks):
      f.write('%s %s %d\n' % (true_key[0], predicted_key[0], rank))
  return np.int32(0)


def _make_logging_ops(true_keys, predicted_keys, ranks, log_dir):
  """tf.metrics-compatible ops for saving and logging results."""
  all_true_keys = []
  all_predicted_keys = []
  all_ranks = []

  def _extend_keys(true_batch_keys, predicted_batch_keys, batch_ranks):
    all_true_keys.extend(true_batch_keys)
    all_predicted_keys.extend(predicted_batch_keys)
    all_ranks.extend(batch_ranks)
    return np.int32(0)

  update_op = tf.py_func(_extend_keys, [true_keys, predicted_keys, ranks],
                         [tf.int32])[0]

  def _write_log_to_file(global_step):
    return _log_predictions(all_true_keys, all_predicted_keys, all_ranks,
                            global_step, log_dir)

  value_op = tf.py_func(_write_log_to_file,
                        [tf.train.get_or_create_global_step()], [tf.int32])[0]

  return (value_op, update_op)


def library_match_accuracy(combined_data,
                           predictor_fn,
                           eval_batch_size,
                           similarity_provider,
                           mass_tolerance,
                           log_dir=None):
  """Compute top-1 library matching accuracy.

  See library_matching() for details of the library matching process.

  Args:
    combined_data: a LibraryMatchingData instance
    predictor_fn: a callable that takes such a dict and returns a predicted
      spectrum.
    eval_batch_size: int for the batch size to use when predicting spectra to
      include in the library.
     similarity_provider: a similarity.SimilarityProvider instance
     mass_tolerance: (Float) library elements are only considered as candidate
       matches if their mass is within this much of the query mass.
     log_dir: (optional) if provided, log predictions here.
  Returns:
    metrics_dict: A dict where each value is a tuple containing an
        Estimator-compatible value_op and update_op.
    library_entry_of_prediction: Float tensor of spectra from library which
        had the best match for each query spectra
    inchikeys: Tensor of strings that are the inchikeys of the spectra in
        library_entry_of_prediction.
  """

  (true_data, predicted_data,
   library_entry_of_prediction, num_library_elements) = library_matching(
       combined_data, predictor_fn, similarity_provider, mass_tolerance,
       eval_batch_size)

  true_inchikeys = true_data[fmap_constants.INCHIKEY]
  predicted_inchikeys = predicted_data[fmap_constants.INCHIKEY]

  best_query_ranks = true_data['rank']['lowest']

  metrics_dict = {}

  if log_dir is not None:
    metrics_dict['prediction_logging'] = _make_logging_ops(
        true_inchikeys, predicted_inchikeys, best_query_ranks, log_dir)

  correct_prediction = tf.equal(true_inchikeys, predicted_inchikeys)
  metrics_dict['library_matching_accuracy'] = tf.metrics.mean(
      correct_prediction)

  metrics_dict[
      'library_matching_fingerprint_jaccard_similarity'] = tf.metrics.mean_iou(
          tf.cast(true_data[FP_NAME_FOR_JACCARD_SIMILARITY] > 0, tf.int32),
          tf.cast(predicted_data[FP_NAME_FOR_JACCARD_SIMILARITY] > 0, tf.int32),
          2)

  metrics_dict['library_match_similarity'] = tf.metrics.mean(
      predicted_data['similarity'])
  metrics_dict['ground_truth_similarity'] = tf.metrics.mean(
      true_data['similarity'])

  metrics_dict['average_query_rank'] = tf.metrics.mean(best_query_ranks)

  for i in [5, 10, 25, 50, 100]:
    metrics_dict['recall@%d' % i] = tf.metrics.mean(best_query_ranks < i)

  metrics_dict['mean_reciprocal_rank'] = tf.metrics.mean(
      tf.pow(tf.to_float(best_query_ranks) + 1, -1))

  avg_query_ranks = true_data['rank']['avg']
  metrics_dict['avg-rank-average_query_rank'] = tf.metrics.mean(avg_query_ranks)

  num_candidates_with_better_scores = true_data['rank']['lowest']
  num_candidates_with_worse_scores = (
      num_library_elements - 1 - true_data['rank']['highest'])
  num_candidates_with_worse_scores = tf.maximum(
      num_candidates_with_worse_scores, 0)

  relative_ranking_position = 0.5 * (
      1 +
      (num_candidates_with_better_scores - num_candidates_with_worse_scores) /
      (num_library_elements - 1))
  metrics_dict['relative_ranking_position'] = tf.metrics.mean(
      relative_ranking_position)

  for i in [5, 10, 25, 50, 100]:
    metrics_dict['avg-rank-recall@%d' %
                 i] = tf.metrics.mean(avg_query_ranks < i)

  return (metrics_dict, library_entry_of_prediction,
          predicted_data[fmap_constants.INCHIKEY])
