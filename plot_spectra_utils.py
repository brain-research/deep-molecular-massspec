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
"""Functions to add an evaluation metric that generates spectra plots."""
from __future__ import print_function

import json
import os

import dataset_setup_constants as ds_constants
import mass_spec_constants as ms_constants
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PilImage
import six
import tensorflow as tf

IMAGE_SUBDIR_FOR_SPECTRA_PLOTS = 'images'

SPECTRA_PLOT_BACKGROUND_COLOR = 'white'
SPECTRA_PLOT_FIGURE_SIZE = (10, 10)
SPECTRA_PLOT_GRID_COLOR = 'black'
SPECTRA_PLOT_TRUE_SPECTRA_COLOR = 'blue'
SPECTRA_PLOT_PREDICTED_SPECTRA_COLOR = 'red'
SPECTRA_PLOT_PEAK_LOC_LIMIT = ms_constants.MAX_PEAK_LOC
SPECTRA_PLOT_MZ_MAX_OFFSET = 10
SPECTRA_PLOT_INTENSITY_LIMIT = 1200
SPECTRA_PLOT_DPI = 300
SPECTRA_PLOT_BAR_LINE_WIDTH = 0.8
SPECTRA_PLOT_BAR_GRID_LINE_WIDTH = 0.1
SPECTRA_PLOT_ACTUAL_SPECTRA_LEGEND_TEXT = 'True Mass Spectrum'
SPECTRA_PLOT_PREDICTED_SPECTRA_LEGEND_TEXT = 'Predicted Mass Spectrum'
SPECTRA_PLOT_QUERY_SPECTRA_LEGEND_TEXT = 'Query Mass Spectrum'
SPECTRA_PLOT_LIBRARY_MATCH_SPECTRA_LEGEND_TEXT = 'Library Matched Mass Spectrum'
SPECTRA_PLOT_X_AXIS_LABEL = 'mass/charge ratio'
SPECTRA_PLOT_Y_AXIS_LABEL = 'relative intensity'
SPECTRA_PLOT_PLACE_LEGEND_ABOVE_CHART_KWARGS = {'ncol': 2}
SPECTRA_PLOT_IMAGE_DIR_NAME = 'images'
SPECTRA_PLOT_DIMENSIONS_RGB = (3000, 3000, 3)
FIGURES_TO_SUMMARIZE_PER_BATCH = 2
MAX_VALUE_OF_TRUE_SPECTRA = 999.


class PlotModeKeys(object):
  """Helper class containing the two supported plotting modes.

  The following keys are defined:
  PREDICTED_SPECTRUM : For plotting the spectrum predicted by the algorithm
      against the true spectrum.
  LIBRARY_MATCHED_SPECTRUM : For plotting the spectrum that was the closest
      match to the true spectrum against the true spectrum.
  """
  PREDICTED_SPECTRUM = 'predicted_spectrum'
  LIBRARY_MATCHED_SPECTRUM = 'library_match_spectrum'


def name_plot_file(mode, query_inchikey, matched_inchikey=None,
                   file_type='png'):
  """Generates name for spectra plot files."""
  if mode == PlotModeKeys.PREDICTED_SPECTRUM:
    return '{}.{}'.format(query_inchikey, file_type)
  elif mode == PlotModeKeys.LIBRARY_MATCHED_SPECTRUM:
    return '{}_matched_to_{}.{}'.format(query_inchikey, matched_inchikey,
                                        file_type)


def name_metric(mode, inchikey):
  return '{}_plot_{}'.format(mode, inchikey)


def plot_true_and_predicted_spectra(
    true_dense_spectra,
    generated_dense_spectra,
    plot_mode_key=PlotModeKeys.PREDICTED_SPECTRUM,
    output_filename='',
    rescale_mz_axis=False):
  """Generates a plot comparing a true and predicted mass spec spectra.

  If output_filename given, saves a png file of the spectra, with the
  true spectrum above and predicted spectrum below.

  Args:
    true_dense_spectra : np.array representing the true mass spectra
    generated_dense_spectra : np.array representing the predicted mass spectra
    plot_mode_key: a PlotModeKeys instance
    output_filename : str path for saving generated image.
    rescale_mz_axis: Setting to rescale m/z axis according to highest m/z peak
        location.

  Returns:
    np.array of the bits of the generated matplotlib plot.
  """

  if not rescale_mz_axis:
    x_array = np.arange(SPECTRA_PLOT_PEAK_LOC_LIMIT)
    bar_width = SPECTRA_PLOT_BAR_LINE_WIDTH
    mz_max = SPECTRA_PLOT_PEAK_LOC_LIMIT
  else:
    mz_max = max(
        max(np.nonzero(true_dense_spectra)[0]),
        max(np.nonzero(generated_dense_spectra)[0]))
    if mz_max + SPECTRA_PLOT_MZ_MAX_OFFSET < ms_constants.MAX_PEAK_LOC:
      mz_max += SPECTRA_PLOT_MZ_MAX_OFFSET
    else:
      mz_max = ms_constants.MAX_PEAK_LOC
    x_array = np.arange(mz_max)
    true_dense_spectra = true_dense_spectra[:mz_max]
    generated_dense_spectra = generated_dense_spectra[:mz_max]
    bar_width = SPECTRA_PLOT_BAR_LINE_WIDTH * mz_max / ms_constants.MAX_PEAK_LOC

  figure = plt.figure(figsize=SPECTRA_PLOT_FIGURE_SIZE, dpi=300)

  # Adding extra subplot so both plots have common x-axis and y-axis labels
  ax_main = figure.add_subplot(111, frameon=False)
  ax_main.tick_params(
      labelcolor='none', top='off', bottom='off', left='off', right='off')

  ax_main.set_xlabel(SPECTRA_PLOT_X_AXIS_LABEL)
  ax_main.set_ylabel(SPECTRA_PLOT_Y_AXIS_LABEL)

  if six.PY2:
    ax_top = figure.add_subplot(211, axisbg=SPECTRA_PLOT_BACKGROUND_COLOR)
  else:
    ax_top = figure.add_subplot(211, facecolor=SPECTRA_PLOT_BACKGROUND_COLOR)

  bar_top = ax_top.bar(
      x_array,
      true_dense_spectra,
      bar_width,
      color=SPECTRA_PLOT_TRUE_SPECTRA_COLOR,
      edgecolor=SPECTRA_PLOT_TRUE_SPECTRA_COLOR,
  )

  ax_top.set_ylim((0, SPECTRA_PLOT_INTENSITY_LIMIT))
  plt.setp(ax_top.get_xticklabels(), visible=False)
  ax_top.grid(
      color=SPECTRA_PLOT_GRID_COLOR, linewidth=SPECTRA_PLOT_BAR_GRID_LINE_WIDTH)

  if six.PY2:
    ax_bottom = figure.add_subplot(212, axisbg=SPECTRA_PLOT_BACKGROUND_COLOR)
  else:
    ax_bottom = figure.add_subplot(212, facecolor=SPECTRA_PLOT_BACKGROUND_COLOR)
  figure.subplots_adjust(hspace=0.0)

  bar_bottom = ax_bottom.bar(
      x_array,
      generated_dense_spectra,
      bar_width,
      color=SPECTRA_PLOT_PREDICTED_SPECTRA_COLOR,
      edgecolor=SPECTRA_PLOT_PREDICTED_SPECTRA_COLOR,
  )

  # Invert the direction of y-axis ticks for bottom graph.
  ax_bottom.set_ylim((SPECTRA_PLOT_INTENSITY_LIMIT, 0))

  ax_bottom.set_xlim(0, mz_max)
  # Remove overlapping 0's from middle of y-axis
  yticks_bottom = ax_bottom.yaxis.get_major_ticks()
  yticks_bottom[0].label1.set_visible(False)

  ax_bottom.grid(
      color=SPECTRA_PLOT_GRID_COLOR, linewidth=SPECTRA_PLOT_BAR_GRID_LINE_WIDTH)

  for ax in [ax_top, ax_bottom]:
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', left='off')
    ax.tick_params(axis='y', which='minor', right='off')

  ax_top.tick_params(axis='x', which='minor', top='off')

  if plot_mode_key == PlotModeKeys.PREDICTED_SPECTRUM:
    ax_top.legend((bar_top, bar_bottom),
                  (SPECTRA_PLOT_ACTUAL_SPECTRA_LEGEND_TEXT,
                   SPECTRA_PLOT_PREDICTED_SPECTRA_LEGEND_TEXT),
                  **SPECTRA_PLOT_PLACE_LEGEND_ABOVE_CHART_KWARGS)
  elif plot_mode_key == PlotModeKeys.LIBRARY_MATCHED_SPECTRUM:
    ax_top.legend((bar_top, bar_bottom),
                  (SPECTRA_PLOT_QUERY_SPECTRA_LEGEND_TEXT,
                   SPECTRA_PLOT_LIBRARY_MATCH_SPECTRA_LEGEND_TEXT),
                  **SPECTRA_PLOT_PLACE_LEGEND_ABOVE_CHART_KWARGS)

  figure.canvas.draw()
  data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')

  try:
    data = np.reshape(data, SPECTRA_PLOT_DIMENSIONS_RGB)
  except ValueError:
    raise ValueError(
        'The shape of the np.array generated from the data does '
        'not match the values in '
        'SPECTRA_PLOT_DIMENSIONS_RGB : {}'.format(SPECTRA_PLOT_DIMENSIONS_RGB))

  if output_filename:
    # We can't call  plt.savefig(output_filename) because plt does not
    # communicate with the filesystem through gfile. In some scenarios, this
    # will prevent us from writing out data. Instead, we use PIL to help us
    # efficiently save the nparray of the image as a png file.
    if not output_filename.endswith('.png') or output_filename.endswith('.eps'):
      output_filename += '.png'

    with tf.gfile.GFile(output_filename, 'wb') as out:
      image = PilImage.fromarray(data).convert('RGB')
      image.save(out, dpi=(SPECTRA_PLOT_DPI, SPECTRA_PLOT_DPI))

  tf.logging.info('Shape of spectra plot data {} '.format(np.shape(data)))

  plt.close(figure)

  return data


def make_plot(inchikey,
              plot_mode_key,
              update_img_flag,
              inchikeys_batch,
              true_spectra_batch,
              predictions,
              image_directory=None,
              library_match_inchikeys=None):
  """Makes plots comparing the true and predicted spectra in a dataset.

  This function only performs the expensive step of generating the spectrum
  plot if the target inchikey is in the current batch.

  Args:
    inchikey: Inchikey of query that we want to make plots with.
    plot_mode_key: A PlotModeKeys instance.
    update_img_flag: Boolean flag for whether to generate a spectra plot
    inchikeys_batch: inchikeys from the current batch
    true_spectra_batch: np.array of all the true spectra from the current batch.
    predictions: np.array of all predicted spectra from the current batch.
    image_directory: Location to save image directory, if set.
    library_match_inchikeys: np.array of strings, corresponding to inchikeys
      that were the best matched from the library inchikey task.

  Returns:
    if update_img_flag: np.array
          [see return value of plot_true_and_predicted_spectra]
    Otherwise, returns a zero np.array of shape SPECTRA_PLOT_DIMENSIONS_RGB.
    Also saves a file at image_directory if this value is non-zero.
  Raises:
    ValueError: library_match_inchikeys needs to be set if given image_directory
        and using PlotModeKeys.LIBRARY_MATCHED_SPECTRUM.
  """
  if update_img_flag:
    flattened_inchikeys_batch = [ikey[0].strip() for ikey in inchikeys_batch]
    inchikey_idx = flattened_inchikeys_batch.index(inchikey)
    predictions = predictions / np.amax(
        predictions, axis=1, keepdims=True) * MAX_VALUE_OF_TRUE_SPECTRA
    predicted_spectra_to_plot = predictions[inchikey_idx, :]
    true_spectra_to_plot = true_spectra_batch[inchikey_idx, :]
    if image_directory:
      if plot_mode_key == PlotModeKeys.PREDICTED_SPECTRUM:
        img_filename = name_plot_file(plot_mode_key, inchikey)
      elif plot_mode_key == PlotModeKeys.LIBRARY_MATCHED_SPECTRUM:
        best_library_match_inchikey = library_match_inchikeys[inchikey_idx, :]
        img_filename = name_plot_file(plot_mode_key, inchikey,
                                      best_library_match_inchikey[0])

      img_pathname = os.path.join(image_directory, img_filename)
      spectra_arr = plot_true_and_predicted_spectra(true_spectra_to_plot,
                                                    predicted_spectra_to_plot,
                                                    plot_mode_key, img_pathname)
    else:
      spectra_arr = plot_true_and_predicted_spectra(true_spectra_to_plot,
                                                    predicted_spectra_to_plot,
                                                    plot_mode_key)
    return spectra_arr
  else:
    return np.zeros(SPECTRA_PLOT_DIMENSIONS_RGB, dtype=np.uint8)


def spectra_plot_summary_op(inchikey_list,
                            true_spectra,
                            prediction_batch,
                            inchikey_to_plot,
                            plot_mode_key=PlotModeKeys.PREDICTED_SPECTRUM,
                            library_match_inchikeys=None,
                            image_directory=None):
  """Wrapper for plotting mass spectra for labels and predictions.

  Plots predicted and true spectra for a given inchikey. If image_directory is
  set, will save the plots as files in addition to making the image summary.

  Args:
    inchikey_list : tf Tensor of inchikey strings for a batch
    true_spectra : tf Tensor array with true spectra for a batch
    prediction_batch: tf Tensor array of predicted spectra for a single batch.
    inchikey_to_plot: string InChI key contained in test set (but perhaps not in
      a particular batch).
    plot_mode_key: A PlotModeKeys instance.
    library_match_inchikeys: tf Tensor of strings corresponding to the inchikeys
      top match from the library matching task.
    image_directory: string of dir name to save plots

  Returns:
     tf.summary.image of the operation, and an update operator indicating if the
     summary has been updated or not.
  """

  def _should_update_image(inchikeys_batch):
    """Tests whether to indicate if target inchikey is in batch."""
    flattened_inchikeys_batch = [ikey[0].strip() for ikey in inchikeys_batch]
    return inchikey_to_plot in flattened_inchikeys_batch

  metric_namescope = 'spectrum_{}_plot_{}'.format(plot_mode_key,
                                                  inchikey_to_plot)
  spectra_variable_name = 'spectrum_{}_plot_{}'.format(plot_mode_key,
                                                       inchikey_to_plot)
  with tf.name_scope(metric_namescope):
    # Whether the inchikey_to_plot is in the current batch.
    update_image_bool = tf.py_func(_should_update_image, [inchikey_list],
                                   tf.bool)

    if plot_mode_key == PlotModeKeys.LIBRARY_MATCHED_SPECTRUM:
      spectra_plot = tf.py_func(make_plot, [
          inchikey_to_plot, plot_mode_key, update_image_bool, inchikey_list,
          true_spectra, prediction_batch, image_directory,
          library_match_inchikeys
      ], tf.uint8)
    elif plot_mode_key == PlotModeKeys.PREDICTED_SPECTRUM:
      spectra_plot = tf.py_func(make_plot, [
          inchikey_to_plot, plot_mode_key, update_image_bool, inchikey_list,
          true_spectra, prediction_batch, image_directory
      ], tf.uint8)

    # Container for the plot. this value will only be assigned to something
    # new if the target inchikey is in the input batch.
    spectra_plot_variable = tf.get_local_variable(
        spectra_variable_name,
        shape=((1,) + SPECTRA_PLOT_DIMENSIONS_RGB),
        initializer=tf.constant_initializer(128),
        dtype=tf.uint8)

    # A function that add the spectra plot as metric.
    def update_function():
      assign_op = spectra_plot_variable.assign(spectra_plot[tf.newaxis, ...])
      with tf.control_dependencies([assign_op]):
        return tf.identity(spectra_plot_variable)

    # We only want to update the metric if the inchikey_to_plot
    # is in the batch. update_image_bool serves as a flag to tf.cond
    # to use the real update function if inchikey_to_plot is in the batch
    # and a fake one if not.
    final_spectra_plot = tf.cond(update_image_bool,
                                 update_function, lambda: spectra_plot_variable)

    update_op = final_spectra_plot

    return (tf.summary.image(
        spectra_variable_name, spectra_plot_variable,
        collections=None), update_op)


def inchikeys_for_plotting(dataset_config_file, num_inchikeys_to_read,
                           eval_batch_size):
  """Return inchikeys from spectrum prediction data file.

  Selects one inchikey per eval batch for plotting. This will avoid the
  threading issue seen at evaluation time.

  Args:
    dataset_config_file: dataset configuration file for experiment. Contains
      filename of spectrum prediction inchikey text file.
    num_inchikeys_to_read: Number of inchikeys to use for plotting
    eval_batch_size: Number of inchikeys to skip before appending the next
      inchikey from the text file.

  Returns:
    list [num_inchikeys_to_read] containing inchikey strings.
  """
  dataset_config_file_dir = os.path.split(dataset_config_file)[0]
  with tf.gfile.Open(dataset_config_file, 'r') as f:
    line = f.read()
    filenames = json.loads(line)
    test_inchikey_list_name = os.path.splitext(filenames[
        ds_constants.SPECTRUM_PREDICTION_TEST_KEY][0])[0] + '.inchikey.txt'

  inchikey_list_for_plotting = []

  with tf.gfile.Open(
      os.path.join(dataset_config_file_dir, test_inchikey_list_name)) as f:
    for line_idx, line in enumerate(f):
      if line_idx % eval_batch_size == 0:
        inchikey_list_for_plotting.append(line.strip('\n'))
      if len(inchikey_list_for_plotting) == num_inchikeys_to_read:
        break

  if len(inchikey_list_for_plotting) < num_inchikeys_to_read:
    tf.logging.warn('Dataset specified by {} has fewer than'
                    '{} inchikeys. Returning {} for plotting'.format(
                        dataset_config_file,
                        num_inchikeys_to_read * eval_batch_size,
                        len(inchikey_list_for_plotting)))
  return inchikey_list_for_plotting
