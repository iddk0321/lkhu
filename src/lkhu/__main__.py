"""``python -m lkhu`` entry point.

Used so that, e.g. on daemon auto-start, lkhu runs via the current interpreter
without relying on PATH.
"""

from lkhu.cli import app

if __name__ == "__main__":
    app()
