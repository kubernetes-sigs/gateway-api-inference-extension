#!/usr/bin/env python3
"""
Build Worker - Watches for trained models and compiles them to TreeLite bundles.

This is the BUILD phase of the Fit/Build split architecture:
- Watches /work/staging/ for READY markers
- Loads native model formats (JSON/txt)
- Compiles to TreeLite .so files
- Creates publishable bundles
- Publishes via existing bundle mechanism
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Configure logging
log_handlers = [logging.StreamHandler(sys.stdout)]

# Add file handler if /work exists (in container), otherwise skip (local dev)
if Path("/work").exists():
    log_handlers.append(logging.FileHandler("/work/build.log"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - BUILD - %(levelname)s - %(message)s", handlers=log_handlers
)

# Import bundle infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.bundle_constants import TPOT_TREELITE_FILENAME, TTFT_TREELITE_FILENAME
from training.bundle_integration import BundleModelManager


class BuildWorker:
    """
    Watches staging directory and builds publishable bundles from trained models.
    """

    def __init__(
        self,
        staging_dir: str = "/work/staging",
        publish_dir: str = "/work/publish",
        bundle_dir: str = None,
        current_symlink: str = None,
        max_bundles: int = 5,
    ):
        self.staging_dir = Path(staging_dir)
        self.publish_dir = Path(publish_dir)

        # Initialize bundle manager (reuses existing infrastructure)
        if bundle_dir is None:
            bundle_dir = os.getenv("BUNDLE_DIR", "/models/bundles")
        if current_symlink is None:
            current_symlink = os.getenv("BUNDLE_CURRENT_SYMLINK", "/models/current")

        self.bundle_manager = BundleModelManager(
            bundle_dir=bundle_dir, current_symlink=current_symlink, max_bundles=max_bundles
        )

        logging.info("BuildWorker initialized:")
        logging.info(f"  Staging: {self.staging_dir}")
        logging.info(f"  Publish: {self.publish_dir}")
        logging.info(f"  Bundles: {bundle_dir}")

        # Ensure directories exist
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.publish_dir.mkdir(parents=True, exist_ok=True)

    def watch_and_build(self, poll_interval: float = 5.0):
        """
        Main loop: watch for READY markers and trigger builds.
        """
        logging.info(f"Starting watch loop (poll_interval={poll_interval}s)")

        processed_runs = set()  # Track completed runs

        while True:
            try:
                # Scan for READY markers
                ready_markers = list(self.staging_dir.glob("*/READY"))

                for ready_path in ready_markers:
                    run_dir = ready_path.parent
                    run_id = run_dir.name

                    if run_id in processed_runs:
                        continue  # Already processed

                    logging.info(f"Found READY marker for run_id={run_id[:8]}")

                    # Acquire lock to prevent double-build
                    lock_path = run_dir / ".build_lock"
                    try:
                        # Try to create lock file (atomic)
                        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(lock_fd)
                    except FileExistsError:
                        logging.info(f"Run {run_id[:8]} already being built by another process")
                        continue

                    try:
                        # Build the bundle
                        success = self.build_bundle(run_dir)

                        if success:
                            processed_runs.add(run_id)
                            # Write PUBLISHED marker
                            published_path = run_dir / "PUBLISHED"
                            with open(published_path, "w") as f:
                                f.write(f"{datetime.now(UTC).isoformat()}\n")
                            logging.info(f"✅ Published bundle for run_id={run_id[:8]}")
                        else:
                            logging.error(f"❌ Build failed for run_id={run_id[:8]}")

                    except Exception as e:
                        logging.error(f"Error building run_id={run_id[:8]}: {e}", exc_info=True)

                    finally:
                        # Release lock
                        try:
                            os.unlink(lock_path)
                        except:
                            pass

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                logging.info("Build worker shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in watch loop: {e}", exc_info=True)
                time.sleep(poll_interval)

    def build_bundle(self, run_dir: Path) -> bool:
        """
        Build a bundle from staging artifacts.

        Steps:
        1. Load and validate fit_manifest.json
        2. Compile models to TreeLite (if needed)
        3. Copy artifacts to bundle directory
        4. Create bundle_manifest.json
        5. Publish bundle

        Returns:
            True if successful, False otherwise
        """
        run_id = run_dir.name
        logging.info(f"Building bundle for run_id={run_id[:8]}")

        # Load fit manifest
        fit_manifest_path = run_dir / "fit_manifest.json"
        if not fit_manifest_path.exists():
            logging.error(f"Missing fit_manifest.json in {run_dir}")
            return False

        with open(fit_manifest_path) as f:
            fit_manifest = json.load(f)

        # Validate artifact hashes
        if not self.validate_artifacts(run_dir, fit_manifest):
            logging.error(f"Artifact validation failed for run_id={run_id[:8]}")
            return False

        # Create bundle directory
        bundle_id = fit_manifest["bundle_id"]
        bundle_path = Path(self.bundle_manager.bundle_dir) / bundle_id
        bundle_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Created bundle directory: {bundle_path}")

        # Copy non-model artifacts first
        for artifact_name, artifact_info in fit_manifest["artifacts"].items():
            if artifact_name in ["ttft_model", "tpot_model"]:
                continue  # Handle models separately (need compilation)

            src_path = run_dir / artifact_info["path"]
            dst_path = bundle_path / artifact_info["path"]

            shutil.copy2(src_path, dst_path)
            logging.info(f"Copied {artifact_name}: {artifact_info['path']}")

        # Compile models to TreeLite (always required for XGBoost and LightGBM)
        logging.info(f"Starting TreeLite compilation for {fit_manifest['model_type']}")

        if not self.compile_models(run_dir, bundle_path, fit_manifest):
            logging.error(f"TreeLite compilation failed for run_id={run_id[:8]}")
            return False

        # Create bundle manifest
        bundle_manifest = self.create_bundle_manifest(bundle_path, fit_manifest)

        bundle_manifest_path = bundle_path / "manifest.json"
        with open(bundle_manifest_path, "w") as f:
            json.dump(bundle_manifest, f, indent=2)

        logging.info(f"Created bundle manifest: {bundle_manifest_path}")

        # Publish bundle (atomic symlink update)
        self.publish_bundle(bundle_id, bundle_path)

        return True

    def validate_artifacts(self, run_dir: Path, fit_manifest: dict[str, Any]) -> bool:
        """Validate that all artifacts exist and hashes match."""
        for artifact_name, artifact_info in fit_manifest["artifacts"].items():
            artifact_path = run_dir / artifact_info["path"]

            if not artifact_path.exists():
                logging.error(f"Missing artifact: {artifact_path}")
                return False

            # Compute hash
            computed_hash = self.compute_file_hash(artifact_path)
            expected_hash = artifact_info["hash"]

            if computed_hash != expected_hash:
                logging.error(
                    f"Hash mismatch for {artifact_name}: " f"expected {expected_hash[:8]}, got {computed_hash[:8]}"
                )
                return False

            logging.debug(f"Validated {artifact_name}: hash={computed_hash[:8]}")

        return True

    def compile_models(self, run_dir: Path, bundle_path: Path, fit_manifest: dict[str, Any]) -> bool:
        """
        Compile native model formats to TreeLite .so files.
        """
        model_type = fit_manifest["model_type"]

        # Import TreeLite libraries
        try:
            import tl2cgen
            import treelite
        except ImportError as e:
            logging.error(f"TreeLite not available: {e}")
            return False

        # Compile TTFT
        if "ttft_model" in fit_manifest["artifacts"]:
            ttft_src = run_dir / fit_manifest["artifacts"]["ttft_model"]["path"]
            ttft_dst = bundle_path / TTFT_TREELITE_FILENAME

            if not self.compile_single_model(ttft_src, ttft_dst, model_type, "TTFT"):
                return False

        # Compile TPOT
        if "tpot_model" in fit_manifest["artifacts"]:
            tpot_src = run_dir / fit_manifest["artifacts"]["tpot_model"]["path"]
            tpot_dst = bundle_path / TPOT_TREELITE_FILENAME

            if not self.compile_single_model(tpot_src, tpot_dst, model_type, "TPOT"):
                return False

        return True

    def compile_single_model(self, model_path: Path, output_path: Path, model_type: str, model_name: str) -> bool:
        """
        Compile a single model to TreeLite .so.
        """
        import tempfile

        import tl2cgen
        import treelite

        logging.info(f"Compiling {model_name} from {model_path}")

        try:
            # Load model based on type
            if model_type == "xgboost":
                # Load XGBoost JSON
                tl_model = treelite.frontend.load_xgboost_model(str(model_path))
                logging.info(f"Loaded XGBoost model: {tl_model.num_tree} trees")

            elif model_type == "lightgbm":
                # Load LightGBM txt
                tl_model = treelite.frontend.load_lightgbm_model(str(model_path))
                logging.info(f"Loaded LightGBM model: {tl_model.num_tree} trees")

            else:
                logging.error(f"Unsupported model type for TreeLite: {model_type}")
                return False

            # Compile to .so
            # Use temp file to avoid partial writes
            with tempfile.NamedTemporaryFile(suffix=".so", delete=False, dir=output_path.parent) as tmp:
                tmp_path = tmp.name

            logging.info(f"Compiling {model_name} to {tmp_path}")
            tl2cgen.export_lib(
                model=tl_model, toolchain="gcc", libpath=tmp_path, params={"parallel_comp": 8}, verbose=True
            )

            # Atomic rename
            os.replace(tmp_path, output_path)

            file_size = output_path.stat().st_size
            logging.info(f"✓ Compiled {model_name}: {output_path} ({file_size:,} bytes)")

            return True

        except Exception as e:
            logging.error(f"Compilation failed for {model_name}: {e}", exc_info=True)
            return False

    def create_bundle_manifest(self, bundle_path: Path, fit_manifest: dict[str, Any]) -> dict[str, Any]:
        """
        Create final bundle manifest with all file hashes.
        """
        manifest = {
            "bundle_id": fit_manifest["bundle_id"],
            "version": "1.0",  # Bundle format version
            "created_at": datetime.now(UTC).isoformat(),
            "model_type": fit_manifest["model_type"],
            "quantile": fit_manifest["quantile"],
            "training_samples": fit_manifest["training_samples"],
            "test_samples": fit_manifest["test_samples"],
            "state": "published",  # Bundle state - always "published" after build completes
            "files": {},
            "toolchain": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
            },
        }

        # Add TreeLite version if available
        try:
            import tl2cgen
            import treelite

            manifest["toolchain"]["treelite_version"] = treelite.__version__
            manifest["toolchain"]["tl2cgen_version"] = tl2cgen.__version__
        except:
            pass

        # Compute hashes for all files in bundle
        for file_path in bundle_path.glob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                file_hash = self.compute_file_hash(file_path)
                file_size = file_path.stat().st_size

                manifest["files"][file_path.name] = {"hash": file_hash, "size_bytes": file_size}

        return manifest

    def publish_bundle(self, bundle_id: str, bundle_path: Path):
        """
        Publish bundle by updating the 'current' symlink atomically.
        """
        current_symlink = Path(self.bundle_manager.current_symlink)

        # Create temporary symlink
        temp_symlink = current_symlink.parent / f".current.tmp.{os.getpid()}"

        # Point temp symlink to new bundle
        if temp_symlink.exists() or temp_symlink.is_symlink():
            temp_symlink.unlink()

        temp_symlink.symlink_to(bundle_path)

        # Atomic rename
        temp_symlink.replace(current_symlink)

        logging.info(f"Published bundle via symlink: {current_symlink} -> {bundle_path}")

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def main():
    """Entry point for build worker."""
    worker = BuildWorker()
    worker.watch_and_build()


if __name__ == "__main__":
    main()
