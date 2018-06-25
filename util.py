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
"""Some general-purpose helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _get_ckpt_from_path(path):
  ckpt = tf.train.latest_checkpoint(path)
  if ckpt is None:
    raise ValueError('No checkpoint found in %s' % path)
  tf.logging.info('Reading from checkpoint %s', ckpt)
  return ckpt


def run_graph_and_process_results(ops_to_fetch,
                                  model_checkpoint_path,
                                  process_fetched_values_fn,
                                  feed_dict=None,
                                  logging_frequency=10):
  """Run a graph repeatedly and use the fetched values.

  Args:
    ops_to_fetch: a single Tensor or nested structure of Tensors. The graph will
      be run, and the below callables will be called, until a
      tf.errors.OutOfRangeError is caught. This is thrown when a tf.data.Dataset
      runs out of data.
    model_checkpoint_path: Path to model checkpoint. If a directory, the most
      recent model checkpoint in this directory will be used.
    process_fetched_values_fn: A callable, potentially with side-effects, that
      takes as input the output of sess.run(ops_to_fetch).
    feed_dict: a feed_dict to be included in sess.run calls.
    logging_frequency: after this many batches have been processed, a logging
    message will be printed.
  """
  ckpt = _get_ckpt_from_path(model_checkpoint_path)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    saver.restore(sess, ckpt)
    counter = 0
    while True:
      try:
        fetched_values = sess.run(ops_to_fetch, feed_dict=feed_dict)
        process_fetched_values_fn(fetched_values)
        counter += 1
        if counter % logging_frequency == 0:
          tf.logging.info('Total examples processed so far: %d', counter)
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished processing data. Processed %d batches',
                        counter)
        break


def map_predictor(input_op, predictor_fn, sub_batch_size):
  """Wrapper for tf.map_fn to do batched computation within each map step."""

  num_elements = tf.contrib.framework.nest.flatten(input_op)[0].shape[0].value

  # Only chop the batch dim into sub-batches if the input data is big.
  if num_elements < sub_batch_size:
    return predictor_fn(input_op)

  pad_amount = -num_elements % sub_batch_size

  def reshape(tensor):
    """Reshape into batches of sub-batches."""
    pad_shape = tensor.shape.as_list()
    pad_shape[0] = pad_amount
    padding = tf.zeros(shape=pad_shape, dtype=tensor.dtype)
    tensor = tf.concat([tensor, padding], axis=0)
    if tensor.shape[0].value % sub_batch_size != 0:
      raise ValueError('Incorrent padding size: %d does not '
                       'divide %d' % (sub_batch_size, tensor.shape[0].value))

    shape = tensor.shape.as_list()
    output_shape = [-1, sub_batch_size] + shape[1:]
    return tf.reshape(tensor, shape=output_shape)

  reshaped_inputs = tf.contrib.framework.nest.map_structure(reshape, input_op)

  mapped_prediction = tf.map_fn(
      predictor_fn,
      reshaped_inputs,
      parallel_iterations=1,
      back_prop=False,
      name=None,
      dtype=tf.float32)

  output_shape = [-1] + mapped_prediction.shape.as_list()[2:]
  reshaped_output = tf.reshape(mapped_prediction, shape=output_shape)

  # If padding was required for the input data, strip off the output of the
  # predictor on this padding.
  if pad_amount > 0:
    reshaped_output = reshaped_output[0:(-pad_amount), ...]

  return reshaped_output


def get_static_shape_without_adding_ops(inputs, fn):
  """Get the shape of fn(inputs) without adding ops to the default graph.

  Operationally equivalent to fn(inputs).shape.as_list(), except that no
  ops are added to the default graph.

  In order to get the shape of fn(inputs) without adding ops to the graph
  we make a new graph, make placeholders with the right shape, construct
  fn(placeholders) in that graph, get the shape, and then delete the graph.

  Note that using this function may have unintended consequences if fn() has
  side effects.

  Args:
    inputs: a (nested) structure where the leaf elements are either Tensors or
      None.
    fn: a function that can be applied to inputs and returns a single Tensor.
  Returns:
    a python list containing the static shape of fn(inputs).

  """
  g = tf.Graph()
  with g.as_default():
    def make_placeholder(tensor):
      if tensor is None:
        return None
      else:
        return tf.placeholder(shape=tensor.shape, dtype=tensor.dtype)

    placeholders = tf.contrib.framework.nest.map_structure(make_placeholder,
                                                           inputs)
    output_shape = fn(placeholders).shape.as_list()

  del g
  return output_shape


def value_op_with_initializer(value_op_fn, init_op_fn):
  """Make value_op that gets set by idempotent init_op on first invocation."""

  init_has_been_run = tf.get_local_variable(
      'has_been_run',
      initializer=np.zeros(shape=(), dtype=np.bool),
      dtype=tf.bool)

  value_op = value_op_fn()

  def run_init_and_toggle():
    init_op = init_op_fn(value_op)

    with tf.control_dependencies([init_op]):
      assign_op = init_has_been_run.assign(True)

    with tf.control_dependencies([assign_op]):
      return tf.identity(value_op)

  return tf.cond(init_has_been_run, lambda: value_op, run_init_and_toggle)


def scatter_by_anchor_indices(anchor_indices, data, index_shift):
  """Shift data such that it is indexed relative to anchor_indices.

  For each row of the data array, we flip it horizontally and then shift it
  so that the output at (anchor_index + index_shift) is the leftmost column
  of the input. Namely:

  output[i][j] = data[i][anchor_indices[i] - j + index_shift]

  Args:
    anchor_indices: [batch_size] int Tensor or np array
    data: [batch_size, num_columns]: float Tensor or np array
    index_shift: int
  Returns:
    [batch_size, num_columns] Tensor
  """
  anchor_indices = tf.convert_to_tensor(anchor_indices)
  data = tf.convert_to_tensor(data)

  num_data_columns = data.shape[-1].value
  indices = np.arange(num_data_columns)[np.newaxis, ...]
  shifted_indices = anchor_indices[..., tf.newaxis] - indices + index_shift
  valid_indices = shifted_indices >= 0

  batch_size = tf.shape(data)[0]

  batch_indices = tf.tile(
      tf.range(batch_size)[..., tf.newaxis], [1, num_data_columns])
  shifted_indices += batch_indices * num_data_columns

  shifted_indices = tf.reshape(shifted_indices, [-1])
  num_elements = tf.shape(data)[0] * tf.shape(data)[1]
  row_indices = tf.range(num_elements)
  stacked_indices = tf.stack([row_indices, shifted_indices], axis=1)

  lower_batch_boundaries = tf.reshape(batch_indices * num_data_columns, [-1])
  upper_batch_boundaries = tf.reshape(((batch_indices + 1) * num_data_columns),
                                      [-1])
  valid_indices = tf.logical_and(shifted_indices >= lower_batch_boundaries,
                                 shifted_indices < upper_batch_boundaries)
  stacked_indices = tf.boolean_mask(
      stacked_indices,
      valid_indices,
  )

  dense_shape = tf.cast(tf.tile(num_elements[..., tf.newaxis], [2]), tf.int64)

  scattering_matrix = tf.SparseTensor(
      indices=tf.cast(stacked_indices, tf.int64),
      values=tf.ones_like(stacked_indices[:, 0], dtype=data.dtype),
      dense_shape=dense_shape)

  flattened_data = tf.reshape(data, [-1])[..., tf.newaxis]
  flattened_output = tf.sparse_tensor_dense_matmul(
      scattering_matrix,
      flattened_data,
      adjoint_a=False,
      adjoint_b=False,
      name=None)

  return tf.reshape(
      tf.transpose(flattened_output, [0, 1]), [-1, num_data_columns])
