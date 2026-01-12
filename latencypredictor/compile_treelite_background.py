#!/usr/bin/env python3
"""
Background TreeLite compilation worker.
This script is launched as a subprocess to compile TreeLite models without blocking the main server.
"""
import sys
import os
import logging
import joblib

# Configure logging to write to a separate file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/treelite_compilation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    import tl2cgen
    import treelite
    import treelite.sklearn
except ImportError as e:
    logging.error(f"Failed to import TreeLite dependencies: {e}")
    sys.exit(1)


def compile_model(model_path: str, output_path: str, model_name: str, versioned_path: str = None):
    """
    Load a joblib model, compile it to TreeLite, and save to output_path.

    Args:
        model_path: Path to .joblib model file
        output_path: Path where compiled .so file should be saved (legacy path for compatibility)
        model_name: Name for logging (e.g., "TTFT" or "TPOT")
        versioned_path: Optional versioned path for runtime updates
    """
    try:
        logging.info(f"Background compilation started for {model_name}")
        logging.info(f"Loading model from {model_path}")

        # Load the model
        model = joblib.load(model_path)
        booster = model.get_booster()

        # Convert to TreeLite format
        logging.info(f"Converting {model_name} to TreeLite format")
        tl_model = treelite.frontend.from_xgboost(booster)
        logging.info(f"{model_name} model converted (num_trees: {tl_model.num_tree})")

        # Compile to shared library (this is the blocking part)
        # Use unique temp file to avoid race conditions between concurrent compilations
        import tempfile
        import time
        tmp_fd, tmp_output = tempfile.mkstemp(suffix='.so', prefix=f'{model_name}_', dir=os.path.dirname(output_path))
        os.close(tmp_fd)  # Close the file descriptor, tl2cgen will write to the path

        logging.info(f"Compiling {model_name} TreeLite model to {tmp_output}")
        tl2cgen.export_lib(
            model=tl_model,
            toolchain='gcc',
            libpath=tmp_output,
            params={'parallel_comp': 8},
            verbose=True
        )

        # Atomic rename to legacy path (for backward compatibility)
        os.replace(tmp_output, output_path)
        logging.info(f"✓ {model_name} TreeLite model successfully compiled and exported to {output_path}")

        # Also save to versioned path if provided (for runtime updates)
        if versioned_path:
            import shutil
            os.makedirs(os.path.dirname(versioned_path), exist_ok=True)
            shutil.copy2(output_path, versioned_path)
            logging.info(f"✓ {model_name} TreeLite model also saved to versioned path: {versioned_path}")

    except Exception as e:
        logging.error(f"Error compiling {model_name} TreeLite model: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print(f"Usage: {sys.argv[0]} <model_path> <output_path> <model_name> [versioned_path]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    model_name = sys.argv[3]
    versioned_path = sys.argv[4] if len(sys.argv) == 5 else None

    compile_model(model_path, output_path, model_name, versioned_path)
