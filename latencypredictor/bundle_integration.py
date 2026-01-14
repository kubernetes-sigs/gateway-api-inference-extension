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

This module provides integration between the new ModelBundle architecture and the existing
path-based model management system. It allows for gradual migration while maintaining
backward compatibility.

Key features:
- Dual-mode operation: bundle mode (new) and legacy mode (old)
- Atomic model exports with completion tracking
- Background compilation tracking for TreeLite
- Single source of truth for model states
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

from model_bundle import ModelBundle, BundleRegistry, BundleState


class CompilationStatus(str, Enum):
    """Status of background TreeLite compilation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BundleModelManager:
    """
    Manager for model bundles (always enabled for atomic model distribution).

    This class provides modern bundle-based model distribution with atomic operations,
    versioning, and immutability. It maintains backward compatibility with legacy paths
    for prediction servers that haven't migrated yet.

    Features:
    - Atomic model exports (temp dir + atomic rename)
    - Background compilation tracking with completion files
    - Single source of truth via bundle manifest
    - Automatic cleanup of old bundles
    - Backward compatible with legacy paths
    """

    def __init__(
        self,
        bundle_dir: str,
        current_symlink: str,
        max_bundles: int = 5,
        use_bundles: bool = True,
        legacy_paths: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the bundle manager.

        Args:
            bundle_dir: Directory to store bundles
            current_symlink: Path to symlink pointing to active bundle
            max_bundles: Maximum number of bundles to keep (older ones deleted)
            use_bundles: Whether to use bundle mode (True) or legacy mode (False)
            legacy_paths: Dict mapping model names to legacy file paths
        """
        self.bundle_dir = Path(bundle_dir)
        self.current_symlink = Path(current_symlink)
        self.max_bundles = max_bundles
        self.use_bundles = use_bundles
        self.legacy_paths = legacy_paths or {}

        # Initialize bundle registry if in bundle mode
        if self.use_bundles:
            self.bundle_dir.mkdir(parents=True, exist_ok=True)
            self.registry = BundleRegistry(str(self.bundle_dir), max_bundles)
            logging.info(f"BundleModelManager initialized in BUNDLE mode: {self.bundle_dir}")
        else:
            logging.info("BundleModelManager initialized in LEGACY mode")
            # Ensure legacy paths exist
            for path in self.legacy_paths.values():
                Path(path).parent.mkdir(parents=True, exist_ok=True)

    def start_training(
        self,
        model_name: str,
        model_type: str,
        quantile: float,
        use_treelite: bool
    ) -> Optional[ModelBundle]:
        """
        Start a new model training session.

        In bundle mode: Creates a new bundle in TRAINING state.
        In legacy mode: No-op, returns None.

        Args:
            model_name: Name of the model (ttft or tpot)
            model_type: Type of model (xgboost, lightgbm, etc.)
            quantile: Target quantile for predictions
            use_treelite: Whether TreeLite compilation is enabled

        Returns:
            ModelBundle in TRAINING state, or None in legacy mode
        """
        if not self.use_bundles:
            return None

        bundle = self.registry.create_bundle(model_type, quantile, use_treelite)
        logging.info(f"Started training for {model_name}: bundle_id={bundle.manifest.bundle_id}")
        return bundle

    def save_model_file(
        self,
        bundle: Optional[ModelBundle],
        file_type: str,
        source_path: str,
        legacy_path: Optional[str] = None
    ) -> bool:
        """
        Save a model file atomically.

        In bundle mode: Adds file to bundle with checksum verification.
        In legacy mode: Copies to legacy path using atomic rename.

        Args:
            bundle: ModelBundle to add file to (None in legacy mode)
            file_type: Type of file (e.g., "ttft_model", "ttft_treelite", "ttft_conformal")
            source_path: Path to source file to copy
            legacy_path: Path to copy to in legacy mode

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.use_bundles and bundle:
                # Bundle mode: add file with checksum
                bundle.add_file(file_type, source_path)
                logging.debug(f"Added {file_type} to bundle {bundle.manifest.bundle_id}")
                return True
            else:
                # Legacy mode: atomic copy
                if not legacy_path:
                    logging.error(f"Legacy path required for {file_type} in legacy mode")
                    return False

                dest = Path(legacy_path)
                dest.parent.mkdir(parents=True, exist_ok=True)

                # Atomic copy via temp file
                tmp_path = dest.with_suffix('.tmp')
                shutil.copy2(source_path, tmp_path)
                os.replace(tmp_path, dest)  # Atomic rename

                logging.debug(f"Saved {file_type} to {legacy_path}")
                return True
        except Exception as e:
            logging.error(f"Error saving {file_type}: {e}", exc_info=True)
            return False

    def start_compilation(
        self,
        bundle: Optional[ModelBundle],
        model_name: str,
        process: subprocess.Popen
    ):
        """
        Track a background TreeLite compilation process.

        In bundle mode: Transitions bundle to COMPILING state and creates tracking file.
        In legacy mode: No-op.

        Args:
            bundle: ModelBundle being compiled
            model_name: Name of model being compiled (ttft or tpot)
            process: Subprocess handle for compilation
        """
        if not self.use_bundles or not bundle:
            return

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
        bundle: Optional[ModelBundle],
        model_name: str
    ) -> Optional[CompilationStatus]:
        """
        Check status of background compilation.

        Args:
            bundle: ModelBundle to check
            model_name: Name of model (ttft or tpot)

        Returns:
            CompilationStatus or None if not found/not in bundle mode
        """
        if not self.use_bundles or not bundle:
            return None

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
        bundle: Optional[ModelBundle],
        model_name: str,
        treelite_path: str,
        success: bool,
        error_msg: Optional[str] = None
    ):
        """
        Mark background compilation as complete.

        In bundle mode: Updates status file and adds .so file to bundle if successful.
        In legacy mode: No-op.

        Args:
            bundle: ModelBundle that was compiled
            model_name: Name of model (ttft or tpot)
            treelite_path: Path to compiled .so file
            success: Whether compilation succeeded
            error_msg: Error message if compilation failed
        """
        if not self.use_bundles or not bundle:
            return

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
            self.save_model_file(bundle, f"{model_name}_treelite", treelite_path)
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
        bundle: Optional[ModelBundle],
        training_samples: Dict[str, int],
        test_samples: Dict[str, int]
    ) -> bool:
        """
        Finalize and publish a bundle.

        In bundle mode: Transitions to READY, updates metadata, publishes via symlink.
        In legacy mode: No-op, returns True.

        Args:
            bundle: ModelBundle to finalize
            training_samples: Dict of training sample counts per bucket
            test_samples: Dict of test sample counts per bucket

        Returns:
            True if successful, False otherwise
        """
        if not self.use_bundles or not bundle:
            return True

        try:
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
            return True
        except Exception as e:
            logging.error(f"Error finalizing bundle: {e}", exc_info=True)
            return False

    def get_active_bundle(self) -> Optional[ModelBundle]:
        """
        Get the currently active bundle.

        Returns:
            Active ModelBundle or None if not in bundle mode or no active bundle
        """
        if not self.use_bundles:
            return None

        return self.registry.get_active_bundle(str(self.current_symlink))

    def list_bundles(self) -> List[ModelBundle]:
        """
        List all bundles (sorted by creation time, newest first).

        Returns:
            List of ModelBundles or empty list if not in bundle mode
        """
        if not self.use_bundles:
            return []

        return self.registry.list_bundles()

    def get_model_path(self, file_type: str) -> Optional[str]:
        """
        Get the current path to a model file.

        In bundle mode: Returns path within active bundle.
        In legacy mode: Returns legacy path.

        Args:
            file_type: Type of file (e.g., "ttft_model", "ttft_treelite")

        Returns:
            Absolute path to file, or None if not found
        """
        if self.use_bundles:
            bundle = self.get_active_bundle()
            if bundle and file_type in bundle.manifest.files:
                return str(bundle.path / file_type)
            return None
        else:
            return self.legacy_paths.get(file_type)
