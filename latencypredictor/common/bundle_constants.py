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
"""

# Bundle file names (used in both training and prediction servers)
# Note: All extensions included for complete consistency
TTFT_MODEL_FILENAME = "ttft_model.joblib"         # joblib serialized model
TPOT_MODEL_FILENAME = "tpot_model.joblib"         # joblib serialized model
TTFT_SCALER_FILENAME = "ttft_scaler.joblib"       # joblib serialized scaler
TPOT_SCALER_FILENAME = "tpot_scaler.joblib"       # joblib serialized scaler
TTFT_TREELITE_FILENAME = "ttft_treelite.so"       # compiled shared library
TPOT_TREELITE_FILENAME = "tpot_treelite.so"       # compiled shared library
TTFT_CONFORMAL_FILENAME = "ttft_conformal.json"   # JSON conformal calibration
TPOT_CONFORMAL_FILENAME = "tpot_conformal.json"   # JSON conformal calibration
