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

"""Constants for creating dataset for mass spectrometry experiment."""

from typing import NamedTuple

# List of components of the main NIST library and the replicates library.
# The main NIST library is divided into a train/validation/test set
# The replicates library is divided into validation/test sets.
MAINLIB_TRAIN_BASENAME = 'mainlib_train'
MAINLIB_VALIDATION_BASENAME = 'mainlib_validation'
MAINLIB_TEST_BASENAME = 'mainlib_test'
REPLICATES_TRAIN_BASENAME = 'replicates_train'
REPLICATES_VALIDATION_BASENAME = 'replicates_validation'
REPLICATES_TEST_BASENAME = 'replicates_test'

# Key names of main datasets required for each experiment.
SPECTRUM_PREDICTION_TRAIN_KEY = 'SPECTRUM_PREDICTION_TRAIN'
SPECTRUM_PREDICTION_TEST_KEY = 'SPECTRUM_PREDICTION_TEST'
LIBRARY_MATCHING_OBSERVED_KEY = 'LIBRARY_MATCHING_OBSERVED'
LIBRARY_MATCHING_PREDICTED_KEY = 'LIBRARY_MATCHING_PREDICTED'
LIBRARY_MATCHING_QUERY_KEY = 'LIBRARY_MATCHING_QUERY'

TRAINING_SPECTRA_ARRAY_KEY = MAINLIB_TRAIN_BASENAME + '_spectra_library_file'


class ExperimentSetup(
    NamedTuple('ExperimentSetup',
               [('json_name', str), ('data_to_get_from_mainlib', list),
                ('data_to_get_from_replicates', list),
                ('experiment_setup_dataset_dict', dict)])):
  """Stores information related to the various experiment setups.

  Attributes:
    json_nane : name of the json file to store locations of the datasets
    data_to_get_from_mainlib: List of dataset keys to grab from the mainlib
        library.
    data_to_get_from_replicates: List of dataset keys to grab from the
        replicates library.
    experiment_setup_dataset_dict: Dict which matches the experiment keys to
        lists of the component datasets, matching the basename keys above.
  """

  def __new__(cls, json_name, data_to_get_from_mainlib,
              data_to_get_from_replicates, experiment_setup_dataset_dict):
    assert (experiment_setup_dataset_dict[LIBRARY_MATCHING_QUERY_KEY] ==
            experiment_setup_dataset_dict[SPECTRUM_PREDICTION_TEST_KEY]), (
                'In json {}, library query list did not match'
                ' spectrum prediction list.'.format(json_name))
    assert (experiment_setup_dataset_dict[SPECTRUM_PREDICTION_TRAIN_KEY] ==
            [MAINLIB_TRAIN_BASENAME]), (
                'In json {}, spectra prediction dataset is not mainlib_train,'
                ' which is currently not supported.'.format(json_name))
    return super(ExperimentSetup, cls).__new__(
        cls, json_name, data_to_get_from_mainlib, data_to_get_from_replicates,
        experiment_setup_dataset_dict)


# Experiment setups:
QUERY_MAINLIB_VAL_PRED_MAINLIB_VAL = ExperimentSetup(
    'query_mainlib_val_predicted_mainlib_val.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY,
        SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_QUERY_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY,
        LIBRARY_MATCHING_PREDICTED_KEY,
    ], [], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [MAINLIB_VALIDATION_BASENAME],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_TRAIN_BASENAME, MAINLIB_TEST_BASENAME,
            REPLICATES_TRAIN_BASENAME, REPLICATES_VALIDATION_BASENAME,
            REPLICATES_TEST_BASENAME
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [MAINLIB_VALIDATION_BASENAME],
        LIBRARY_MATCHING_QUERY_KEY: [MAINLIB_VALIDATION_BASENAME],
    })

QUERY_REPLICATES_VAL_PRED_REPLICATES_VAL = ExperimentSetup(
    'query_replicates_val_predicted_replicates_val.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY,
        SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY,
        LIBRARY_MATCHING_PREDICTED_KEY,
    ], [LIBRARY_MATCHING_QUERY_KEY], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [REPLICATES_VALIDATION_BASENAME],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_TRAIN_BASENAME, MAINLIB_TEST_BASENAME,
            MAINLIB_VALIDATION_BASENAME, REPLICATES_TRAIN_BASENAME,
            REPLICATES_TEST_BASENAME
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [REPLICATES_VALIDATION_BASENAME],
        LIBRARY_MATCHING_QUERY_KEY: [REPLICATES_VALIDATION_BASENAME],
    })

QUERY_REPLICATES_TEST_PRED_REPLICATES_TEST = ExperimentSetup(
    'query_replicates_test_predicted_replicates_test.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY,
        SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY,
        LIBRARY_MATCHING_PREDICTED_KEY,
    ], [LIBRARY_MATCHING_QUERY_KEY], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [REPLICATES_TEST_BASENAME],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_TRAIN_BASENAME, MAINLIB_TEST_BASENAME,
            REPLICATES_TRAIN_BASENAME, MAINLIB_VALIDATION_BASENAME,
            REPLICATES_VALIDATION_BASENAME
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [REPLICATES_TEST_BASENAME],
        LIBRARY_MATCHING_QUERY_KEY: [REPLICATES_TEST_BASENAME],
    })

QUERY_REPLICATES_VAL_PRED_NONE = ExperimentSetup(
    'query_replicates_val_predicted_none.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY,
        SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY,
        LIBRARY_MATCHING_PREDICTED_KEY,
    ], [LIBRARY_MATCHING_QUERY_KEY], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [REPLICATES_VALIDATION_BASENAME],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_TRAIN_BASENAME, MAINLIB_TEST_BASENAME,
            MAINLIB_VALIDATION_BASENAME, REPLICATES_TRAIN_BASENAME,
            REPLICATES_TEST_BASENAME, REPLICATES_VALIDATION_BASENAME
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [],
        LIBRARY_MATCHING_QUERY_KEY: [REPLICATES_VALIDATION_BASENAME],
    })

QUERY_REPLICATES_TEST_PRED_NONE = ExperimentSetup(
    'query_replicates_test_predicted_none.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY,
        SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY,
        LIBRARY_MATCHING_PREDICTED_KEY,
    ], [LIBRARY_MATCHING_QUERY_KEY], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [REPLICATES_TEST_BASENAME],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_TRAIN_BASENAME, MAINLIB_TEST_BASENAME,
            MAINLIB_VALIDATION_BASENAME, REPLICATES_TRAIN_BASENAME,
            REPLICATES_VALIDATION_BASENAME, REPLICATES_TEST_BASENAME
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [],
        LIBRARY_MATCHING_QUERY_KEY: [REPLICATES_TEST_BASENAME],
    })

QUERY_REPLICATES_ALL_PRED_NONE = ExperimentSetup(
    'query_replicates_all_predicted_none.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY,
        SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY,
        LIBRARY_MATCHING_PREDICTED_KEY,
    ], [LIBRARY_MATCHING_QUERY_KEY], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [
            REPLICATES_VALIDATION_BASENAME, REPLICATES_TEST_BASENAME
        ],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_TRAIN_BASENAME, MAINLIB_TEST_BASENAME,
            MAINLIB_VALIDATION_BASENAME, REPLICATES_TRAIN_BASENAME,
            REPLICATES_VALIDATION_BASENAME, REPLICATES_TEST_BASENAME
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [],
        LIBRARY_MATCHING_QUERY_KEY: [
            REPLICATES_VALIDATION_BASENAME, REPLICATES_TEST_BASENAME
        ],
    })

# An overfitting setup for sanity checks
QUERY_MAINLIB_TRAIN_PRED_MAINLIB_TRAIN = ExperimentSetup(
    'query_mainlib_train_predicted_mainlib_train.json', [
        SPECTRUM_PREDICTION_TRAIN_KEY, SPECTRUM_PREDICTION_TEST_KEY,
        LIBRARY_MATCHING_OBSERVED_KEY, LIBRARY_MATCHING_PREDICTED_KEY,
        LIBRARY_MATCHING_QUERY_KEY
    ], [], {
        SPECTRUM_PREDICTION_TRAIN_KEY: [MAINLIB_TRAIN_BASENAME],
        SPECTRUM_PREDICTION_TEST_KEY: [MAINLIB_TRAIN_BASENAME],
        LIBRARY_MATCHING_OBSERVED_KEY: [
            MAINLIB_VALIDATION_BASENAME,
            MAINLIB_TEST_BASENAME,
            REPLICATES_TRAIN_BASENAME,
            REPLICATES_VALIDATION_BASENAME,
            REPLICATES_TEST_BASENAME,
        ],
        LIBRARY_MATCHING_PREDICTED_KEY: [MAINLIB_TRAIN_BASENAME],
        LIBRARY_MATCHING_QUERY_KEY: [MAINLIB_TRAIN_BASENAME],
    })

EXPERIMENT_SETUPS_LIST = [
    QUERY_MAINLIB_VAL_PRED_MAINLIB_VAL,
    QUERY_REPLICATES_VAL_PRED_REPLICATES_VAL,
    QUERY_REPLICATES_TEST_PRED_REPLICATES_TEST,
    QUERY_REPLICATES_VAL_PRED_NONE,
    QUERY_REPLICATES_TEST_PRED_NONE,
    QUERY_MAINLIB_TRAIN_PRED_MAINLIB_TRAIN,
]
