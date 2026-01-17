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
Bundle Integration Layer

This module provides high-level integration for the ModelBundle architecture,
enabling atomic model distribution with versioning and immutability.

Key features:
- Atomic model exports with completion tracking
- Background compilation tracking for TreeLite
- Single source of truth for model states via bundle manifests
- Automatic cleanup of old bundles
"""

import os
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum

from .model_bundle import ModelBundle, BundleRegistry, BundleState
from common.bundle_constants import TTFT_TREELITE_FILENAME, TPOT_TREELITE_FILENAME


class CompilationStatus(str, Enum):
    """Status of background TreeLite compilation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BundleModelManager:
    """
    Manager for model bundles using atomic operations and versioning.

    This class provides bundle-based model distribution with atomic operations,
    versioning, and immutability.

    Features:
    - Atomic model exports (temp dir + atomic rename)
    - Background compilation tracking with completion files
    - Single source of truth via bundle manifest
    - Automatic cleanup of old bundles
    """

    def __init__(
        self,
        bundle_dir: str,
        current_symlink: str,
        max_bundles: int = 5
    ):
        """
        Initialize the bundle manager.

        Args:
            bundle_dir: Directory to store bundles
            current_symlink: Path to symlink pointing to active bundle
            max_bundles: Maximum number of bundles to keep (older ones deleted)
        """
        self.bundle_dir = Path(bundle_dir)
        self.current_symlink = Path(current_symlink)
        self.max_bundles = max_bundles

        # Initialize bundle registry
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.registry = BundleRegistry(str(self.bundle_dir), max_bundles)
        logging.info(f"BundleModelManager initialized: {self.bundle_dir}")

    def start_training(
        self,
        model_name: str,
        model_type: str,
        quantile: float,
        use_treelite: bool
    ) -> ModelBundle:
        """
        Start a new model training session.

        Creates a new bundle in TRAINING state.

        Args:
            model_name: Name of the model (ttft or tpot)
            model_type: Type of model (xgboost, lightgbm, etc.)
            quantile: Target quantile for predictions
            use_treelite: Whether TreeLite compilation is enabled

        Returns:
            ModelBundle in TRAINING state
        """
        bundle = self.registry.create_bundle(model_type, quantile, use_treelite)
        logging.info(f"Started training for {model_name}: bundle_id={bundle.manifest.bundle_id}")
        return bundle

    def save_model_file(
        self,
        bundle: ModelBundle,
        file_type: str,
        source_path: str
    ) -> bool:
        """
        Save a model file atomically to the bundle.

        Adds file to bundle with checksum verification.

        Args:
            bundle: ModelBundle to add file to
            file_type: Type of file (e.g., "ttft_model", "ttft_treelite", "ttft_conformal")
            source_path: Path to source file to copy

        Returns:
            True if successful, False otherwise
        """
        try:
            bundle.add_file(file_type, source_path)
            logging.debug(f"Added {file_type} to bundle {bundle.manifest.bundle_id}")
            return True
        except Exception as e:
            logging.error(f"Error saving {file_type}: {e}", exc_info=True)
            return False

    def start_compilation(
        self,
        bundle: ModelBundle,
        model_name: str,
        process: subprocess.Popen
    ):
        """
        Track a background TreeLite compilation process.

        Transitions bundle to COMPILING state and creates tracking file.

        Args:
            bundle: ModelBundle being compiled
            model_name: Name of model being compiled (ttft or tpot)
            process: Subprocess handle for compilation
        """
        # Transition to COMPILING state
        bundle.manifest.transition_state(
            BundleState.COMPILING,
            f"TreeLite compilation started for {model_name} (PID {process.pid})"
        )

        # Create compilation tracking file
        status_file = bundle.path / f"{model_name}_compilation_status.json"
        import json
        with open(status_file, 'w') as f:
            json.dump({
                'status': CompilationStatus.RUNNING,
                'pid': process.pid,
                'started_at': datetime.now(timezone.utc).isoformat(),
                'model_name': model_name
            }, f, indent=2)

        logging.info(f"Tracking compilation for {model_name} in bundle {bundle.manifest.bundle_id}")

    def check_compilation_status(
        self,
        bundle: ModelBundle,
        model_name: str
    ) -> Optional[CompilationStatus]:
        """
        Check status of background compilation.

        Args:
            bundle: ModelBundle to check
            model_name: Name of model (ttft or tpot)

        Returns:
            CompilationStatus or None if not found
        """
        status_file = bundle.path / f"{model_name}_compilation_status.json"
        if not status_file.exists():
            return None

        import json
        try:
            with open(status_file) as f:
                data = json.load(f)
            return CompilationStatus(data['status'])
        except Exception as e:
            logging.warning(f"Error reading compilation status: {e}")
            return None

    def mark_compilation_complete(
        self,
        bundle: ModelBundle,
        model_name: str,
        treelite_path: str,
        success: bool,
        error_msg: Optional[str] = None
    ):
        """
        Mark background compilation as complete.

        Updates status file and adds .so file to bundle if successful.

        Args:
            bundle: ModelBundle that was compiled
            model_name: Name of model (ttft or tpot)
            treelite_path: Path to compiled .so file
            success: Whether compilation succeeded
            error_msg: Error message if compilation failed
        """
        status_file = bundle.path / f"{model_name}_compilation_status.json"
        import json

        # Update status file
        status_data = {
            'status': CompilationStatus.COMPLETED if success else CompilationStatus.FAILED,
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'model_name': model_name
        }

        if success:
            # Add .so file to bundle
            treelite_filename = TTFT_TREELITE_FILENAME if model_name == "ttft" else TPOT_TREELITE_FILENAME
            self.save_model_file(bundle, treelite_filename, treelite_path)
            status_data['treelite_path'] = treelite_path
        else:
            status_data['error'] = error_msg

        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

        logging.info(
            f"Compilation {'succeeded' if success else 'failed'} for {model_name} "
            f"in bundle {bundle.manifest.bundle_id}"
        )

    def finalize_bundle(
        self,
        bundle: ModelBundle,
        training_samples: Dict[str, int],
        test_samples: Dict[str, int]
    ):
        """
        Finalize and publish a bundle.

        Transitions to READY, updates metadata, and publishes via atomic symlink update.

        Args:
            bundle: ModelBundle to finalize
            training_samples: Dict of training sample counts per bucket
            test_samples: Dict of test sample counts per bucket

        Raises:
            Exception if finalization fails
        """
        # Update sample counts
        bundle.manifest.training_samples = training_samples
        bundle.manifest.test_samples = test_samples

        # Transition to READY
        bundle.manifest.transition_state(
            BundleState.READY,
            f"All files written and verified. Ready to publish."
        )
        bundle.save_manifest()

        # Publish bundle (atomic symlink update)
        bundle.publish(str(self.current_symlink))

        # Cleanup old bundles
        self.registry.cleanup_old_bundles(str(self.current_symlink))

        logging.info(
            f"Published bundle {bundle.manifest.bundle_id}: "
            f"{sum(training_samples.values())} training samples, "
            f"{sum(test_samples.values())} test samples"
        )

    def get_active_bundle(self) -> Optional[ModelBundle]:
        """
        Get the currently active bundle.

        Returns:
            Active ModelBundle or None if no active bundle
        """
        return self.registry.get_active_bundle(str(self.current_symlink))

    def list_bundles(self) -> List[ModelBundle]:
        """
        List all bundles (sorted by creation time, newest first).

        Returns:
            List of ModelBundles
        """
        return self.registry.list_bundles()

    def get_model_path(self, file_type: str) -> Optional[str]:
        """
        Get the current path to a model file within the active bundle.

        Args:
            file_type: Type of file (e.g., "ttft_model", "ttft_treelite")

        Returns:
            Absolute path to file, or None if not found
        """
        bundle = self.get_active_bundle()
        if bundle and file_type in bundle.manifest.files:
            return str(bundle.path / file_type)
        return None
