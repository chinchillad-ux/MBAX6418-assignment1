"""Entry point for current three-class dashboard browser tests."""
if not __debug__:
    raise SystemExit('Integrity checks require assertions: run without -O/-OO and unset PYTHONOPTIMIZE.')

from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('test_dashboard_three.py')), run_name='__main__')
