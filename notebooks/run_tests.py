# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars python-dateutil scikit-learn pytest pytest-cov

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys

# Add the src directory to sys.path so insurance_cv is importable
# without needing to run pip install -e .
sys.path.insert(0, "/Workspace/insurance-cv-migration/src")

# Verify import works
import insurance_cv
print(f"insurance_cv loaded from: {insurance_cv.__file__}")

# COMMAND ----------
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-cv-migration/tests",
        "-v", "--tb=long",
        "--import-mode=importlib",
        f"--rootdir=/Workspace/insurance-cv-migration",
    ],
    capture_output=True,
    text=True,
    env={**__import__("os").environ, "PYTHONPATH": "/Workspace/insurance-cv-migration/src"},
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-5000:] if len(output) > 5000 else output)
