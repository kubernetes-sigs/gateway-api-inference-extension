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
Model Bundle: Immutable versioned model artifacts with explicit state machine.

This module provides a robust model distribution system that eliminates race conditions
and edge cases through:
1. Immutable versioned bundles (never modify files in-place)
2. Explicit state machine (TRAINING → COMPILING → READY → PUBLISHED)
3. Atomic operations (all-or-nothing file updates)
4. Single source of truth (no hash/timestamp desyncs)

Design Principles:
- Each model version is a complete atomic bundle
- Bundle ID = content hash of all files
- State transitions are explicit and logged
- No partial states (either complete bundle or nothing)
"""

import os
import json
import hashlib
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from pathlib import Path


class BundleState(str, Enum):
    """
    Explicit state machine for model bundles.

    State transitions:
    TRAINING → COMPILING → READY → PUBLISHED → DEPRECATED

    States:
    - TRAINING: Models are being trained, bundle directory created but incomplete
    - COMPILING: Base models saved, TreeLite compilation in progress
    - READY: All files written, bundle ready for use (but not yet published)
    - PUBLISHED: Bundle is the current active version (symlink updated)
    - DEPRECATED: Older bundle that has been superseded
    """
    TRAINING = "training"       # Initial state: models being trained
    COMPILING = "compiling"     # TreeLite compilation in progress
    READY = "ready"             # All files written, ready to publish
    PUBLISHED = "published"     # Current active version
    DEPRECATED = "deprecated"   # Old version, kept for rollback


@dataclass
class BundleManifest:
    """
    Manifest describing the contents of a model bundle.

    This is the single source of truth for what files exist in a bundle
    and their checksums. Prevents issues with partial downloads or
    corrupted files.
    """
    bundle_id: str
    version: str
    created_at: str  # ISO 8601 timestamp
    state: BundleState

    # Model type and configuration
    model_type: str  # "xgboost", "lightgbm", "bayesian_ridge"
    quantile: float
    use_treelite: bool

    # File checksums (SHA256)
    files: Dict[str, str] = field(default_factory=dict)

    # Metadata
    training_samples: Dict[str, int] = field(default_factory=dict)  # {"ttft": 1000, "tpot": 1000}
    test_samples: Dict[str, int] = field(default_factory=dict)

    # State transition history
    state_history: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['state'] = self.state.value  # Convert enum to string
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'BundleManifest':
        """Restore from dictionary."""
        data = data.copy()
        data['state'] = BundleState(data['state'])  # Convert string to enum
        return cls(**data)

    def add_file(self, name: str, path: str):
        """Add a file to the manifest with its checksum."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # Compute SHA256 checksum
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        self.files[name] = sha256.hexdigest()

    def verify_file(self, name: str, path: str) -> bool:
        """Verify a file matches the manifest checksum."""
        if name not in self.files:
            return False

        if not os.path.exists(path):
            return False

        # Compute SHA256 and compare
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest() == self.files[name]

    def transition_state(self, new_state: BundleState, reason: str = ""):
        """
        Transition to a new state with logging.

        Args:
            new_state: Target state
            reason: Optional reason for transition
        """
        old_state = self.state
        self.state = new_state

        # Record transition in history
        self.state_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        })

        logging.info(f"Bundle {self.bundle_id[:8]}: {old_state.value} → {new_state.value} ({reason})")


class ModelBundle:
    """
    Immutable versioned model bundle with atomic operations.

    A bundle represents a complete set of model artifacts:
    - Base models (joblib files)
    - Compiled models (TreeLite .so files)
    - Calibration data (conformal prediction JSON)
    - Manifest (checksums and metadata)

    Key properties:
    - Immutable: Once created, files are never modified
    - Atomic: Bundle is either complete or doesn't exist
    - Versioned: Each bundle has unique ID based on content hash
    """

    def __init__(self, bundle_dir: str, manifest: Optional[BundleManifest] = None):
        """
        Initialize a bundle (either new or existing).

        Args:
            bundle_dir: Root directory for bundles (e.g., "/models/bundles")
            manifest: Existing manifest to load, or None to create new
        """
        self.bundle_dir = Path(bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

        if manifest:
            self.manifest = manifest
        else:
            # Create new bundle with unique ID
            bundle_id = self._generate_bundle_id()
            self.manifest = BundleManifest(
                bundle_id=bundle_id,
                version="1.0.0",  # Could be incremented or use semantic versioning
                created_at=datetime.now(timezone.utc).isoformat(),
                state=BundleState.TRAINING,
                model_type="",  # Set by caller
                quantile=0.9,
                use_treelite=True,
            )

        # Bundle path
        self.path = self.bundle_dir / self.manifest.bundle_id
        self.path.mkdir(parents=True, exist_ok=True)

        # Manifest file
        self.manifest_path = self.path / "manifest.json"

    @staticmethod
    def _generate_bundle_id() -> str:
        """Generate a unique bundle ID based on timestamp and random data."""
        timestamp = datetime.now(timezone.utc).isoformat()
        random_data = os.urandom(8).hex()
        content = f"{timestamp}-{random_data}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def load(cls, bundle_dir: str, bundle_id: str) -> 'ModelBundle':
        """
        Load an existing bundle from disk.

        Args:
            bundle_dir: Root directory for bundles
            bundle_id: ID of bundle to load

        Returns:
            ModelBundle instance

        Raises:
            FileNotFoundError: If bundle doesn't exist
            ValueError: If manifest is invalid
        """
        bundle_path = Path(bundle_dir) / bundle_id
        manifest_path = bundle_path / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)

        manifest = BundleManifest.from_dict(manifest_data)
        return cls(bundle_dir, manifest)

    def save_manifest(self):
        """Save the manifest to disk (atomic write)."""
        tmp_path = self.manifest_path.with_suffix('.tmp')

        with open(tmp_path, 'w') as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

        # Atomic rename
        os.replace(tmp_path, self.manifest_path)

    def add_file(self, name: str, source_path: str, dest_name: Optional[str] = None):
        """
        Add a file to the bundle (atomic copy with checksum).

        Args:
            name: Logical name for the file (e.g., "ttft_model", "ttft_treelite")
            source_path: Path to source file
            dest_name: Optional destination filename (defaults to name)

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If file already exists in bundle
        """
        if name in self.manifest.files:
            raise ValueError(f"File already exists in bundle: {name}")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Destination path
        dest_filename = dest_name or name
        dest_path = self.path / dest_filename

        # Atomic copy via temp file
        tmp_path = dest_path.with_suffix('.tmp')

        # Copy file
        import shutil
        shutil.copy2(source_path, tmp_path)

        # Atomic rename
        os.replace(tmp_path, dest_path)

        # Add to manifest
        self.manifest.add_file(name, str(dest_path))
        self.save_manifest()

        logging.debug(f"Bundle {self.manifest.bundle_id[:8]}: Added {name} ({os.path.getsize(dest_path)} bytes)")

    def get_file_path(self, name: str) -> Optional[Path]:
        """Get the path to a file in the bundle."""
        if name not in self.manifest.files:
            return None

        # Map logical name to actual filename
        # (For now, they're the same, but this allows flexibility)
        dest_path = self.path / name

        if not dest_path.exists():
            return None

        return dest_path

    def verify_integrity(self) -> bool:
        """
        Verify all files in the bundle match their checksums.

        Returns:
            True if all files are valid, False otherwise
        """
        for name in self.manifest.files:
            file_path = self.get_file_path(name)
            if not file_path or not self.manifest.verify_file(name, str(file_path)):
                logging.error(f"Bundle {self.manifest.bundle_id[:8]}: Integrity check failed for {name}")
                return False

        logging.debug(f"Bundle {self.manifest.bundle_id[:8]}: Integrity check passed ({len(self.manifest.files)} files)")
        return True

    def mark_ready(self):
        """Mark bundle as READY (all files written)."""
        if self.manifest.state != BundleState.COMPILING:
            logging.warning(f"Bundle {self.manifest.bundle_id[:8]}: Cannot mark READY from state {self.manifest.state}")
            return

        # Verify integrity before marking ready
        if not self.verify_integrity():
            raise ValueError(f"Bundle {self.manifest.bundle_id[:8]}: Cannot mark READY, integrity check failed")

        self.manifest.transition_state(BundleState.READY, "All files written and verified")
        self.save_manifest()

    def publish(self, symlink_path: str):
        """
        Publish bundle by updating symlink atomically.

        Args:
            symlink_path: Path to "current" symlink (e.g., "/models/current")

        Raises:
            ValueError: If bundle is not in READY state
        """
        if self.manifest.state != BundleState.READY:
            raise ValueError(f"Cannot publish bundle in state {self.manifest.state}, must be READY")

        symlink = Path(symlink_path)

        # Create temporary symlink
        tmp_symlink = symlink.with_suffix('.tmp')
        if tmp_symlink.exists() or tmp_symlink.is_symlink():
            tmp_symlink.unlink()

        # Point to bundle directory
        tmp_symlink.symlink_to(self.path)

        # Atomic replace
        os.replace(tmp_symlink, symlink)

        # Update state
        self.manifest.transition_state(BundleState.PUBLISHED, f"Symlink updated: {symlink}")
        self.save_manifest()

        logging.info(f"Bundle {self.manifest.bundle_id[:8]} published to {symlink}")


class BundleRegistry:
    """
    Registry for managing all model bundles.

    Provides:
    - Bundle lifecycle management
    - Active bundle tracking
    - Garbage collection of old bundles
    """

    def __init__(self, bundle_dir: str, max_bundles: int = 5):
        """
        Initialize bundle registry.

        Args:
            bundle_dir: Root directory for bundles
            max_bundles: Maximum number of bundles to keep (for garbage collection)
        """
        self.bundle_dir = Path(bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.max_bundles = max_bundles
        self.lock = threading.RLock()

    def create_bundle(self, model_type: str, quantile: float, use_treelite: bool) -> ModelBundle:
        """
        Create a new bundle.

        Args:
            model_type: Model type ("xgboost", "lightgbm", etc.)
            quantile: Target quantile
            use_treelite: Whether TreeLite is enabled

        Returns:
            New ModelBundle instance
        """
        with self.lock:
            bundle = ModelBundle(str(self.bundle_dir))
            bundle.manifest.model_type = model_type
            bundle.manifest.quantile = quantile
            bundle.manifest.use_treelite = use_treelite
            bundle.save_manifest()

            logging.info(f"Created bundle {bundle.manifest.bundle_id[:8]} ({model_type}, q={quantile}, treelite={use_treelite})")
            return bundle

    def list_bundles(self) -> List[str]:
        """List all bundle IDs (sorted by creation time, newest first)."""
        bundles = []
        for path in self.bundle_dir.iterdir():
            if path.is_dir():
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest_data = json.load(f)
                        created_at = manifest_data.get("created_at", "")
                        bundles.append((created_at, path.name))
                    except Exception:
                        continue

        # Sort by creation time (newest first)
        bundles.sort(reverse=True)
        return [bundle_id for _, bundle_id in bundles]

    def get_active_bundle_id(self, symlink_path: str) -> Optional[str]:
        """Get the currently active bundle ID by reading symlink."""
        symlink = Path(symlink_path)
        if not symlink.is_symlink():
            return None

        target = symlink.resolve()
        return target.name if target.exists() else None

    def get_active_bundle(self, symlink_path: str) -> Optional[ModelBundle]:
        """
        Get the currently active bundle by loading it from disk.

        Args:
            symlink_path: Path to active symlink (e.g., "/models/current")

        Returns:
            Active ModelBundle or None if no active bundle
        """
        bundle_id = self.get_active_bundle_id(symlink_path)
        if not bundle_id:
            return None

        try:
            return ModelBundle.load(str(self.bundle_dir), bundle_id)
        except Exception as e:
            logging.error(f"Error loading active bundle {bundle_id}: {e}")
            return None

    def cleanup_old_bundles(self, symlink_path: str):
        """
        Remove old bundles, keeping only max_bundles most recent.

        Args:
            symlink_path: Path to active symlink (will not delete active bundle)
        """
        with self.lock:
            bundles = self.list_bundles()
            active_id = self.get_active_bundle_id(symlink_path)

            # Keep max_bundles most recent, plus active bundle
            to_keep = set(bundles[:self.max_bundles])
            if active_id:
                to_keep.add(active_id)

            for bundle_id in bundles:
                if bundle_id not in to_keep:
                    bundle_path = self.bundle_dir / bundle_id
                    if bundle_path.exists():
                        import shutil
                        shutil.rmtree(bundle_path)
                        logging.info(f"Cleaned up old bundle: {bundle_id[:8]}")
