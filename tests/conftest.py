"""Test bootstrap — guarantee tests run offline with no API keys."""

import os
import sys

# Force every phase into its offline / stub path. Don't trust the dev's .env.
os.environ.pop("OPENAI_API_KEY", None)
os.environ["IMAGE_BACKEND"] = "placeholder"

# Make sure the project root is on sys.path even when pytest is run from elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
