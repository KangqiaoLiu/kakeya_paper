"""Reproduce the calculations and figures from this directory's sources."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(relative_path):
    script = ROOT / relative_path
    print(f"Running {relative_path}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-only', action='store_true',
                        help='Use the supplied loss-geometry data to redraw figures.')
    args = parser.parse_args()
    run('fig1/generate_source_data_fig1.py')
    run('fig1/kakeya_fig1.py')
    if not args.figures_only:
        for script in ['run_exploration.py', 'run_validation.py', 'run_realization.py',
                       'run_design_checks.py', 'build_summary.py']:
            run('loss_geometry/' + script)
    run('loss_geometry/make_figures.py')


if __name__ == '__main__':
    main()
