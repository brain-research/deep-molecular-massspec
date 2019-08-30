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
"""Tests for deep_molecular_massspec.library_matching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import feature_map_constants as fmap_constants
import library_matching
import similarity as similarity_lib
import numpy as np
import tensorflow as tf

PREDICTOR_INPUT_KEY = 'INPUT'


class LibraryMatchingTest(tf.test.TestCase):

  def testCosineSimilarityProviderMatching(self):
    """Check correctness for querying the library with a library element."""

    num_examples = 20
    num_trials = 10
    data_dim = 5
    similarity = similarity_lib.CosineSimilarityProvider()
    library = np.float32(np.random.normal(size=(num_examples, data_dim)))
    library = tf.constant(library)
    library = similarity.preprocess_library(library)
    query_idx = tf.placeholder(shape=(), dtype=tf.int32)
    query = library[query_idx][np.newaxis, ...]
    (match_idx_op, match_similarity_op, _, _,
     _) = library_matching._max_similarity_match(library, query, similarity)

    # Use queries that are rows of the library. This means that the maximum
    # cosine similarity is 1.0 and is achieved by the row index of the query
    # in the library.
    with tf.Session() as sess:
      for _ in range(num_trials):
        idx = np.random.randint(0, high=num_examples)
        match_idx, match_similarity = sess.run(
            [match_idx_op, match_similarity_op], feed_dict={query_idx: idx})
        # Fail if the match_idx != idx, and the similarity of match_idx does
        # is not tied with the argmax (which is 1.0 by construction).
        if match_idx != idx:
          self.assertClose(match_similarity, 1.0)

  def testFindQueryPositions(self):
    """Test library_matching._find_query_rank_helper."""

    library_keys = np.array(['a', 'b', 'c', 'd', 'a', 'b'])
    query_keys = np.array(['a', 'b', 'c', 'a', 'b', 'c'])

    similarities = np.array(
        [[3., 4., 6., 0., 0.1, 2.], [1., -1., 5., 3, 2., 1.1],
         [-5., 0., 2., 0.1, 3., 1.], [0.2, 0.4, 0.6, 0.32, 0.3, 0.9],
         [0.2, 0.9, 0.65, 0.18, 0.3, 0.99], [0.8, 0.6, 0.5, 0.4, 0.9, 0.89]])

    (highest_query_ranks, lowest_query_ranks, avg_query_ranks,
     query_similarities) = library_matching._find_query_rank_helper(
         similarities, library_keys, query_keys)

    expected_highest_query_ranks = [4, 5, 1, 5, 1, 4]
    expected_lowest_query_ranks = [2, 3, 1, 4, 0, 4]
    expected_avg_query_ranks = [3, 4, 1, 4.5, 0.5, 4]
    expected_query_similarities = [3., 1.1, 2., 0.3, 0.99, 0.5]

    self.assertAllEqual(expected_highest_query_ranks, highest_query_ranks)
    self.assertAllEqual(expected_lowest_query_ranks, lowest_query_ranks)
    self.assertAllEqual(expected_avg_query_ranks, avg_query_ranks)

    self.assertAllEqual(expected_query_similarities, query_similarities)

  def testInvertPermutation(self):
    """Test library_matching._invert_permutation()."""

    batch_size = 5
    num_trials = 10
    permutation_length = 6

    def _validate_permutation(perm1, perm2):
      ordered_indices = np.arange(perm1.shape[0])
      self.assertAllEqual(perm1[perm2], ordered_indices)
      self.assertAllEqual(perm2[perm1], ordered_indices)

    for _ in range(num_trials):
      perms = np.stack(
          [
              np.random.permutation(permutation_length)
              for _ in range(batch_size)
          ],
          axis=0)

      inverse = library_matching._invert_permutation(perms)

      for j in range(batch_size):
        _validate_permutation(perms[j], inverse[j])

  def np_normalize_rows(self, d):
    return d / np.maximum(
        np.linalg.norm(d, axis=1)[..., np.newaxis], similarity_lib.EPSILON)

  def make_ids(self, num_ids, prefix=''):
    if prefix:
      prefix += '-'

    return [('%s%d' % (prefix, uid)).encode('utf-8') for uid in range(num_ids)]

  def _random_fingerprint(self, num_elements):
    return tf.to_float(tf.random_uniform(shape=(num_elements, 1024)) > 0.5)

  def _package_data(self, ids, spectrum, masses):

    def convert(t):
      if t is None:
        return t
      else:
        return tf.convert_to_tensor(t)

    num_elements = len(ids)
    fingerprints = self._random_fingerprint(num_elements)
    return {
        fmap_constants.DENSE_MASS_SPEC: convert(spectrum),
        fmap_constants.INCHIKEY: convert(ids),
        library_matching.FP_NAME_FOR_JACCARD_SIMILARITY: fingerprints,
        fmap_constants.MOLECULE_WEIGHT: convert(masses)
    }

  def make_x_data(self, num_examples, x_dim):
    return np.float32(np.random.uniform(size=(num_examples, x_dim)))

  def np_library_matching(self, ids_predicted, ids_observed, y_predicted,
                          y_observed, y_query):
    ids_library = np.concatenate([ids_predicted, ids_observed])
    np_library = self.np_normalize_rows(
        np.concatenate([y_predicted, y_observed], axis=0))
    np_similarities = np.dot(np_library, np.transpose(y_query))
    np_predictions = np.argmax(np_similarities, axis=0)
    np_predicted_ids = [ids_library[i] for i in np_predictions]
    return np_predicted_ids

  def perform_matching(self, ids_observed, ids_predicted, ids_query,
                       masses_observed, masses_predicted, masses_query,
                       y_observed, y_query, x_predicted, tf_transform,
                       mass_tolerance):

    query_data = self._package_data(
        ids=ids_query, spectrum=y_query, masses=masses_query)

    predicted_data = self._package_data(
        ids=ids_predicted, spectrum=None, masses=masses_predicted)
    predicted_data[PREDICTOR_INPUT_KEY] = tf.constant(x_predicted)

    observed_data = self._package_data(
        ids=ids_observed, spectrum=y_observed, masses=masses_observed)

    library_matching_data = library_matching.LibraryMatchingData(
        query=query_data, observed=observed_data, predicted=predicted_data)

    predictor_fn = lambda d: tf_transform(d[PREDICTOR_INPUT_KEY])
    similarity = similarity_lib.CosineSimilarityProvider()
    true_data, predicted_data, _, _ = (
        library_matching.library_matching(library_matching_data, predictor_fn,
                                          similarity, mass_tolerance, 10))

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      return sess.run([predicted_data, true_data])

  def tf_vs_np_library_matching_test_helper(self,
                                            num_observed,
                                            num_predicted,
                                            query_source,
                                            num_queries=5):
    """Helper for asserting TF and NP give same library matching output."""

    x_dim = 5

    np_transform = lambda d: np.sqrt(d + 4)
    tf_transform = lambda t: tf.sqrt(t + 4)

    x_observed = self.make_x_data(num_observed, x_dim)
    x_predicted = self.make_x_data(num_predicted, x_dim)

    y_observed = np_transform(x_observed)
    y_predicted = np_transform(x_predicted)

    if query_source == 'random':
      # Use queries from the same generating process as the observed and
      # predicted data.
      x_query = self.make_x_data(num_queries, x_dim)
      y_query = np_transform(x_query)
    elif query_source == 'observed':
      # Copy the observed data to use as queries.
      y_query = y_observed
    elif query_source == 'zero':
      # Use the zero vector as the queries.
      y_query = np.zeros(shape=(num_queries, x_dim), dtype=np.float32)
    else:
      raise ValueError('Invalid query_source: %s' % query_source)

    ids_observed = self.make_ids(num_observed)
    ids_predicted = self.make_ids(num_predicted)
    ids_query = self.make_ids(num_queries)
    masses_observed = np.ones([num_observed, 1], dtype=np.float32)
    masses_predicted = np.ones([num_predicted, 1], dtype=np.float32)
    masses_query = np.ones([num_queries, 1], dtype=np.float32)

    (predicted_data, true_data) = self.perform_matching(
        ids_observed,
        ids_predicted,
        ids_query,
        masses_observed,
        masses_predicted,
        masses_query,
        y_observed,
        y_query,
        x_predicted,
        tf_transform,
        mass_tolerance=3)
    np_predicted_ids = self.np_library_matching(
        ids_predicted, ids_observed, y_predicted, y_observed, y_query)

    # Assert correctness of the ids of the library matches found by TF.
    self.assertAllEqual(np_predicted_ids,
                        predicted_data[fmap_constants.INCHIKEY])

    # Assert correctness of the ground truth ids output extracted by TF.
    self.assertAllEqual(true_data[fmap_constants.INCHIKEY], ids_query)

    # Assert that a query spectrum that is in the observed set should be matched
    # to the corresponding element in the observed set.
    if query_source == 'observed':
      self.assertAllEqual(ids_observed, predicted_data[fmap_constants.INCHIKEY])

  def testLibraryMatchingTFvsNP(self):
    return self.tf_vs_np_library_matching_test_helper(
        num_observed=10, num_predicted=5, query_source='random')

  def testLibraryMatchingTFvsNPZeroObserved(self):
    return self.tf_vs_np_library_matching_test_helper(
        num_observed=0, num_predicted=5, query_source='random')

  def testLibraryMatchingTFvsZeroPredicted(self):
    return self.tf_vs_np_library_matching_test_helper(
        num_observed=10, num_predicted=0, query_source='random')

  def testLibraryMatchingTFvsNPQueriesObserved(self):
    return self.tf_vs_np_library_matching_test_helper(
        num_observed=10,
        num_predicted=5,
        query_source='observed',
        num_queries=10)

  def testLibraryMatchingTFvsNPZeroQueries(self):
    return self.tf_vs_np_library_matching_test_helper(
        num_observed=10, num_predicted=5, query_source='zero')

  def testLibraryMatchingHardcoded(self):
    """Test library_matching using hardcoded values."""

    tf_transform = lambda t: t + 2
    x_predicted = np.array([[1, 1], [-3, -2]], dtype=np.float32)

    y_observed = np.array([[1, 2], [2, 1], [0, 0]], dtype=np.float32)
    y_query = np.array([[2, 5], [2, 1], [-1.5, -1.1]], dtype=np.float32)

    ids_observed = self.make_ids(3, 'obs')
    ids_predicted = self.make_ids(2, 'pred')
    ids_query = np.array([b'obs-0', b'obs-1', b'pred-1'])
    masses_observed = np.ones([3, 1], dtype=np.float32)
    masses_predicted = np.ones([2, 1], dtype=np.float32)
    masses_query = np.ones([3, 1], dtype=np.float32)

    expected_predicted_ids = ids_query.tolist()

    predicted_data, _ = self.perform_matching(
        ids_observed,
        ids_predicted,
        ids_query,
        masses_observed,
        masses_predicted,
        masses_query,
        y_observed,
        y_query,
        x_predicted,
        tf_transform,
        mass_tolerance=3)

    self.assertAllEqual(expected_predicted_ids,
                        predicted_data[fmap_constants.INCHIKEY])

  def testLibraryMatchingHardcodedMassFiltered(self):
    """Test library_matching using hardcoded values when filtering by mass."""

    tf_transform = lambda t: t + 2
    x_predicted = np.array([[1, 1], [-3, -2]], dtype=np.float32)

    y_observed = np.array([[1, 2], [2, 1], [0, 0]], dtype=np.float32)
    y_query = np.array([[2, 5], [2, 1], [-1.5, -1.1]], dtype=np.float32)

    ids_observed = self.make_ids(3, 'obs')
    ids_predicted = self.make_ids(2, 'pred')
    ids_query = np.array([b'pred-0', b'obs-1', b'obs-2'])
    masses_observed = np.ones([3, 1], dtype=np.float32)
    masses_predicted = 2 * np.ones([2, 1], dtype=np.float32)
    masses_query = np.array([3, 1.5, 0], dtype=np.float32)[..., np.newaxis]

    expected_predicted_ids = ids_query.tolist()

    predicted_data, _ = self.perform_matching(
        ids_observed,
        ids_predicted,
        ids_query,
        masses_observed,
        masses_predicted,
        masses_query,
        y_observed,
        y_query,
        x_predicted,
        tf_transform,
        mass_tolerance=1)

    self.assertAllEqual(expected_predicted_ids,
                        predicted_data[fmap_constants.INCHIKEY])

  def testMassFilterRaisesError(self):
    """Test case where mass filtering removes everything."""

    tf_transform = lambda t: t + 2
    x_predicted = np.array([[1, 1], [-3, -2]], dtype=np.float32)

    y_observed = np.array([[1, 2], [2, 1], [0, 0]], dtype=np.float32)
    y_query = np.array([[2, 5]], dtype=np.float32)

    ids_observed = self.make_ids(3, 'obs')
    ids_predicted = self.make_ids(2, 'pred')
    ids_query = np.array(['pred-0'])
    masses_observed = np.ones([3, 1], dtype=np.float32)
    masses_predicted = 2 * np.ones([2, 1], dtype=np.float32)
    masses_query = np.array([5], dtype=np.float32)[..., np.newaxis]

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.perform_matching(
          ids_observed,
          ids_predicted,
          ids_query,
          masses_observed,
          masses_predicted,
          masses_query,
          y_observed,
          y_query,
          x_predicted,
          tf_transform,
          mass_tolerance=1)

  def testLibraryMatchingNoPredictions(self):
    """Test library_matching using hardcoded values with no predicted data."""

    y_observed = np.array([[1, 2], [2, 1], [0, 0]], dtype=np.float32)
    y_query = np.array([[2, 5], [-3, 1], [0, 0]], dtype=np.float32)

    ids_observed = self.make_ids(3)
    ids_query = self.make_ids(3)

    expected_predictions = [b'0', b'2', b'0']

    masses_query = np.ones([3, 1], dtype=np.float32)
    query_data = self._package_data(
        ids=ids_query, spectrum=y_query, masses=masses_query)
    masses_observed = np.ones([3, 1], dtype=np.float32)
    observed_data = self._package_data(
        ids=ids_observed, spectrum=y_observed, masses=masses_observed)
    predicted_data = None
    library_matching_data = library_matching.LibraryMatchingData(
        query=query_data, observed=observed_data, predicted=predicted_data)

    _, predicted_data, _, _ = library_matching.library_matching(
        library_matching_data,
        predictor_fn=None,
        similarity_provider=similarity_lib.CosineSimilarityProvider(),
        mass_tolerance=3.0)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      predictions = sess.run(predicted_data[fmap_constants.INCHIKEY])

    self.assertAllEqual(expected_predictions, predictions)

  def testLibraryMatchingNoObserved(self):
    """Test library_matching using hardcoded values with no observed data."""

    tf_transform = lambda t: t + 2
    x_predicted = np.array([[1, 1], [-3, -2]], dtype=np.float32)
    y_query = np.array([[2, 5], [-3, 1], [0, 0]], dtype=np.float32)

    ids_predicted = self.make_ids(2)
    ids_query = self.make_ids(3)

    expected_predictions = [b'0', b'1', b'0']

    masses_query = np.ones([3, 1], dtype=np.float32)
    query_data = self._package_data(
        ids=ids_query, spectrum=y_query, masses=masses_query)
    masses_predicted = np.ones([2, 1], dtype=np.float32)
    predicted_data = self._package_data(
        ids=ids_predicted, spectrum=None, masses=masses_predicted)
    predicted_data[PREDICTOR_INPUT_KEY] = tf.constant(x_predicted)

    observed_data = None
    library_matching_data = library_matching.LibraryMatchingData(
        query=query_data, observed=observed_data, predicted=predicted_data)

    predictor_fn = lambda d: tf_transform(d[PREDICTOR_INPUT_KEY])

    _, predicted_data, _, _ = library_matching.library_matching(
        library_matching_data,
        predictor_fn=predictor_fn,
        similarity_provider=similarity_lib.CosineSimilarityProvider(),
        mass_tolerance=3.0)

    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      predictions = sess.run(predicted_data[fmap_constants.INCHIKEY])

    self.assertAllEqual(expected_predictions, predictions)


if __name__ == '__main__':
  tf.test.main()
