import unittest,sys,os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
suite=unittest.defaultTestLoader.discover('tests')
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
