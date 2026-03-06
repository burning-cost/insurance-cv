"""
Run insurance-cv test suite on Databricks serverless compute.
Uploads the project source and executes pytest via the Jobs API.
"""

import os
import sys
import time
import base64

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/insurance-cv-migration"
PROJECT_ROOT = "/home/ralph/insurance-cv"

# Files to upload (relative to PROJECT_ROOT)
UPLOAD_PATHS = [
    "src/insurance_cv/__init__.py",
    "src/insurance_cv/splits.py",
    "src/insurance_cv/diagnostics.py",
    "tests/__init__.py",
    "tests/test_splits.py",
    "examples/motor_walkforward.py",
    "pyproject.toml",
    "README.md",
]


def upload_file(local_path: str, remote_path: str) -> None:
    with open(local_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.AUTO,
    )
    print(f"  Uploaded: {remote_path}")


def upload_notebook(local_path: str, remote_path: str) -> None:
    """Upload a Python file as a Databricks SOURCE notebook."""
    with open(local_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode()
    parent = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        overwrite=True,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
    )
    print(f"  Uploaded notebook: {remote_path}")


print("Uploading project files to Databricks workspace...")
for rel_path in UPLOAD_PATHS:
    local = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(local):
        print(f"  SKIP (not found): {local}")
        continue
    remote = f"{WORKSPACE_PATH}/{rel_path}"
    upload_file(local, remote)

notebook_local = os.path.join(PROJECT_ROOT, "notebooks/run_tests.py")
notebook_remote = f"{WORKSPACE_PATH}/notebooks/run_tests"
upload_notebook(notebook_local, notebook_remote)

print("\nSubmitting test job (serverless compute)...")

run = w.jobs.submit(
    run_name="insurance-cv-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            environment_key="default",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_remote,
            ),
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=compute.Environment(),
        )
    ],
)

run_id = run.run_id
print(f"Job run submitted: run_id={run_id}")
print(f"Monitor at: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")

print("\nPolling for completion...")
while True:
    state = w.jobs.get_run(run_id=run_id)
    life_cycle = state.state.life_cycle_state.value
    print(f"  Status: {life_cycle}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(15)

result_state = state.state.result_state
print(f"\nResult: {result_state.value if result_state else 'UNKNOWN'}")

try:
    task_run = state.tasks[0] if state.tasks else None
    if task_run:
        output = w.jobs.get_run_output(run_id=task_run.run_id)
        if output.notebook_output and output.notebook_output.result:
            print("\n--- Notebook output ---")
            print(output.notebook_output.result)
        if output.error:
            print(f"\nError: {output.error}")
        if output.error_trace:
            print(f"\nTrace:\n{output.error_trace}")
except Exception as e:
    print(f"Could not fetch output: {e}")

if result_state and result_state.value == "SUCCESS":
    print("\nTests PASSED.")
    sys.exit(0)
else:
    print("\nTests FAILED.")
    sys.exit(1)
