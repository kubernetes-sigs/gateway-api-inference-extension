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
Lifecycle State Module

Shared lifecycle state tracking for training server and build worker coordination.
Uses a JSON file on a shared volume for inter-process communication.

States:
    ABSENT              - No models exist (cold start, no seed bundle)
    WAITING_FOR_SAMPLES - Collecting training samples, not enough to train yet
    TRAINING            - Training is in progress
    COMPILING           - Build worker is compiling TreeLite models
    READY               - Models are trained and published
    ERROR               - Training or compilation failed
"""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from enum import Enum

DEFAULT_LIFECYCLE_PATH = os.getenv("LIFECYCLE_STATE_PATH", "/work/lifecycle_state.json")


class LifecycleState(str, Enum):
    ABSENT = "ABSENT"
    WAITING_FOR_SAMPLES = "WAITING_FOR_SAMPLES"
    TRAINING = "TRAINING"
    COMPILING = "COMPILING"
    READY = "READY"
    ERROR = "ERROR"


def read_lifecycle_state(path: str = None) -> dict:
    """
    Read the current lifecycle state from the shared JSON file.

    Returns a dict with keys: state, bundle_id, reason, last_error, sample_count, timestamp.
    If the file is missing or unreadable, returns ABSENT defaults.
    """
    if path is None:
        path = DEFAULT_LIFECYCLE_PATH

    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "state": data.get("state", LifecycleState.ABSENT.value),
            "bundle_id": data.get("bundle_id"),
            "reason": data.get("reason"),
            "last_error": data.get("last_error"),
            "sample_count": data.get("sample_count", 0),
            "timestamp": data.get("timestamp"),
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {
            "state": LifecycleState.ABSENT.value,
            "bundle_id": None,
            "reason": None,
            "last_error": None,
            "sample_count": 0,
            "timestamp": None,
        }


def write_lifecycle_state(
    state: LifecycleState,
    reason: str = None,
    bundle_id: str = None,
    last_error: str = None,
    sample_count: int = 0,
    path: str = None,
) -> None:
    """
    Atomically write lifecycle state to the shared JSON file.

    Uses tmpfile + os.replace for atomic writes (no partial reads).
    """
    if path is None:
        path = DEFAULT_LIFECYCLE_PATH

    data = {
        "state": state.value,
        "bundle_id": bundle_id,
        "reason": reason,
        "last_error": last_error,
        "sample_count": sample_count,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Ensure parent directory exists
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Atomic write: write to temp file, then replace
    try:
        fd, tmp_path = tempfile.mkstemp(dir=parent_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        logging.warning(f"Failed to write lifecycle state to {path}: {e}")
