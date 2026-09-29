"""Rebuild the current full-run presentation; the pilot builder is archived."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("build_full_run_data_round.py")), run_name="__main__")
