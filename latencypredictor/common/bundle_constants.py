# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bundle Constants

Shared constants for bundle file naming across training and prediction servers.
This ensures consistency in file naming and eliminates hardcoded strings.

NOTE: Model filenames are now dynamic based on model_type (xgboost vs lightgbm).
Use get_model_filename() helper function instead of static constants where model_type varies.
"""

# TreeLite compiled models (constant across model types)
TTFT_TREELITE_FILENAME = "ttft_treelite.so"  # compiled shared library
TPOT_TREELITE_FILENAME = "tpot_treelite.so"  # compiled shared library

# Conformal prediction weights (constant across model types)
TTFT_CONFORMAL_FILENAME = "ttft_conformal.json"  # JSON conformal calibration
TPOT_CONFORMAL_FILENAME = "tpot_conformal.json"  # JSON conformal calibration


def get_model_filename(model_type: str, model_name: str) -> str:
    """
    Get the native model filename based on model type.

    Args:
        model_type: 'xgboost' or 'lightgbm'
        model_name: 'ttft' or 'tpot'

    Returns:
        Filename with correct extension for the model type

    Examples:
        >>> get_model_filename('xgboost', 'ttft')
        'ttft_model.json'
        >>> get_model_filename('lightgbm', 'tpot')
        'tpot_model.txt'
    """
    if model_type not in ["xgboost", "lightgbm"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'xgboost' or 'lightgbm'")

    if model_name not in ["ttft", "tpot"]:
        raise ValueError(f"Invalid model_name: {model_name}. Must be 'ttft' or 'tpot'")

    extension = "json" if model_type == "xgboost" else "txt"
    return f"{model_name}_model.{extension}"
