"""
Setup and verification script for Kronos Candlestick Foundation Model.

Clones the official Kronos repository (MIT license, AAAI 2026) into `vendor/kronos`
and verifies that the tokeniser and model weights can be loaded from HuggingFace.

Usage:
    python scripts/setup_kronos.py
    python scripts/setup_kronos.py --download-weights
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / "vendor"
KRONOS_DIR = VENDOR_DIR / "kronos"

KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
DEFAULT_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_MODEL_ID = "NeoQuasar/Kronos-small"


def clone_kronos_if_needed() -> bool:
    """Clone Kronos repo into vendor/kronos if not present."""
    if (KRONOS_DIR / "model" / "kronos.py").is_file():
        print(f"[OK] Kronos repository is already present at: {KRONOS_DIR}")
        return True

    print(f"Cloning Kronos repository into {KRONOS_DIR}...")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", KRONOS_REPO_URL, str(KRONOS_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[OK] Successfully cloned Kronos to: {KRONOS_DIR}")
        return True
    except subprocess.CalledProcessError as err:
        print(f"[ERROR] Failed to clone Kronos: {err.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("[ERROR] 'git' command not found. Please install Git or clone manually:", file=sys.stderr)
        print(f"  git clone --depth 1 {KRONOS_REPO_URL} {KRONOS_DIR}", file=sys.stderr)
        return False


def verify_kronos_import() -> bool:
    """Verify that Kronos modules can be imported."""
    if str(KRONOS_DIR) not in sys.path:
        sys.path.insert(0, str(KRONOS_DIR))

    try:
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore[import-untyped]
        print("[OK] Successfully imported Kronos, KronosTokenizer, and KronosPredictor.")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to import Kronos: {exc}", file=sys.stderr)
        print("Ensure dependencies are installed: pip install einops huggingface_hub torch", file=sys.stderr)
        return False


def verify_weights(tokenizer_id: str = DEFAULT_TOKENIZER_ID, model_id: str = DEFAULT_MODEL_ID) -> bool:
    """Check if weights can be loaded from HuggingFace."""
    if str(KRONOS_DIR) not in sys.path:
        sys.path.insert(0, str(KRONOS_DIR))

    try:
        from model import Kronos, KronosTokenizer  # type: ignore[import-untyped]
        print(f"Verifying HuggingFace tokenizer weights: {tokenizer_id}...")
        tok = KronosTokenizer.from_pretrained(tokenizer_id)
        print(f"[OK] Tokenizer loaded successfully: {type(tok).__name__}")

        print(f"Verifying HuggingFace model weights: {model_id}...")
        mod = Kronos.from_pretrained(model_id)
        print(f"[OK] Model loaded successfully: {type(mod).__name__}")
        return True
    except Exception as exc:
        print(f"[WARNING] Could not load weights from HuggingFace: {exc}")
        print("Note: Weights will be automatically downloaded on first inference run.", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup and verify Kronos foundation model vendor directory.")
    parser.add_argument("--download-weights", action="store_true", help="Download and verify pretrained weights from HF")
    args = parser.parse_args()

    print("=" * 70)
    print("  Kronos Candlestick Foundation Model Setup")
    print("=" * 70)

    if not clone_kronos_if_needed():
        return 1

    if not verify_kronos_import():
        return 1

    if args.download_weights:
        verify_weights()

    print("\nKronos is configured and ready for walk-forward evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
