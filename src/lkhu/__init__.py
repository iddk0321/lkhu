"""lkhu (Like Human) — an MCP plugin that gives memory working like the human brain.

It stores and processes memories as latent vectors ("scents") rather than natural
language, reducing token cost.
"""

import os as _os

# faiss links its own OpenMP runtime. In case it loads alongside other OpenMP-using
# libraries, allow the duplicate runtime (defensively) and pin faiss to a single
# thread (core/long_term.py).
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"

__all__ = ["__version__"]
