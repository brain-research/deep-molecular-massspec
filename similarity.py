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

"""Helper functions for similarity computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

import numpy as np
import tensorflow as tf

# When computing cosine similarity, the denominator is constrained to be no
# smaller than this.
EPSILON = 1e-6


class SimilarityProvider(object):
  """Abstract class of helpers for similarity-based library matching."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, hparams=None):
    self.hparams = hparams

  @abc.abstractmethod
  def preprocess_library(self, library):
    """Perform normalization of [num_library_elements, feature_dim] library."""

  @abc.abstractmethod
  def undo_library_preprocessing(self, library):
    """Undo the effect of preprocess_library(), up to a scaling constant."""

  @abc.abstractmethod
  def preprocess_queries(self, queries):
    """Perform normalization of [num_query_elements, feature_dim] queries."""

  @abc.abstractmethod
  def compute_similarity(self, library, queries):
    """Compute [num_library_elements, num_query_elements] similarities."""

  @abc.abstractmethod
  def make_training_loss(self, true_tensor, predicted_tensor):
    """Create training loss that is consistent with the similarity."""


class CosineSimilarityProvider(SimilarityProvider):
  """Cosine similarity."""

  def _normalize_rows(self, tensor):
    return tf.nn.l2_normalize(tensor, axis=1)

  def preprocess_library(self, library):
    return self._normalize_rows(library)

  def undo_library_preprocessing(self, library):
    return library

  def preprocess_queries(self, queries):
    return self._normalize_rows(queries)

  def compute_similarity(self, library, queries):
    similarities = tf.matmul(library, queries, transpose_b=True)
    return tf.transpose(similarities)

  def make_training_loss(self, true_tensor, predicted_tensor):
    return tf.reduce_mean(
        tf.losses.mean_squared_error(true_tensor, predicted_tensor))


class GeneralizedCosineSimilarityProvider(CosineSimilarityProvider):
  """Custom cosine similarity that is popular for massspec matching."""

  def _make_weights(self, tensor):
    num_bins = tensor.shape[1].value
    weights = np.power(np.arange(1, num_bins + 1),
                       self.hparams.mass_power)[np.newaxis, :]
    return weights / np.sum(weights)

  def _normalize_rows(self, tensor):
    if self.hparams.mass_power != 0:
      tensor *= self._make_weights(tensor)

    return super(GeneralizedCosineSimilarityProvider,
                 self)._normalize_rows(tensor)

  def undo_library_preprocessing(self, library):
    return library / self._make_weights(library)

  def compute_similarity(self, library, queries):
    similarities = tf.matmul(library, queries, transpose_b=True)
    return tf.transpose(similarities)

  def make_training_loss(self, true_tensor, predicted_tensor):
    if self.hparams.mass_power != 0:
      weights = self._make_weights(true_tensor)
      weighted_squared_error = weights * tf.square(true_tensor -
                                                   predicted_tensor)
      return tf.reduce_mean(weighted_squared_error)
    else:
      return tf.reduce_mean(
          tf.losses.mean_squared_error(true_tensor, predicted_tensor))


def max_margin_ranking_loss(predictions, target_indices, library,
                            similarity_provider, margin):
  """Max-margin ranking loss.

  loss = (1/batch_size) * sum_i w_i sum_j max(0,
                                          similarities[i, j]
                                          - similarities[i, ti] + margin),
  where similarities = similarity_provider.compute_similarity(library,
                                                              predictions)
  and ti = target_indices[i]. Here, w_i is a weight placed on each element of
  the batch. Without w_i, our loss would be the standard Crammer-Singer
  multiclass svm. Instead, we set w_i so that the total constribution to the
  parameter gradient from each batch element is equal. Therefore, we set w_i
  equal to 1 / (the number of margin violations for element i).

  Args:
    predictions: [batch_size, prediction_dim] float Tensor
    target_indices: [batch_size] int Tensor
    library: [num_library_elements, prediction_dim] constant Tensor
    similarity_provider: a SimilarityProvider instance
    margin: float
  Returns:
    loss
  """
  library = similarity_provider.preprocess_library(library)
  predictions = similarity_provider.preprocess_queries(predictions)
  similarities = similarity_provider.compute_similarity(library, predictions)

  batch_size = tf.shape(predictions)[0]

  target_indices = tf.squeeze(target_indices, axis=1)
  row_indices = tf.range(0, batch_size)
  indices = tf.stack([row_indices, tf.cast(target_indices, tf.int32)], axis=1)
  ground_truth_similarities = tf.gather_nd(similarities, indices)

  margin_violations = tf.nn.relu(-ground_truth_similarities[..., tf.newaxis] +
                                 similarities + margin)

  margin_violators = tf.cast(margin_violations > 0, tf.int32)
  margin_violators_per_batch_element = tf.to_float(
      tf.reduce_sum(margin_violators, axis=1, keep_dims=True))
  margin_violators_per_batch_element = tf.maximum(
      margin_violators_per_batch_element, 1.)
  margin_violators_per_batch_element = tf.stop_gradient(
      margin_violators_per_batch_element)
  tf.summary.scalar('num_margin_violations',
                    tf.reduce_mean(margin_violators_per_batch_element))
  weighted_margin_violations = (
      margin_violations / margin_violators_per_batch_element)
  return tf.reduce_sum(weighted_margin_violations) / tf.maximum(
      tf.to_float(batch_size), 1.)
