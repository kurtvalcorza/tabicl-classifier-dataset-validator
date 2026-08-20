# Make the repo root importable so `import validator` resolves under the `pytest`
# console script (which, unlike `python -m pytest`, does not add the cwd to
# sys.path). pytest imports this conftest and inserts its directory.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
