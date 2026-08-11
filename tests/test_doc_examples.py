"""
Tests for code examples in the documentation.

Each test executes a script from docs/examples/ directly,
so the docs and tests always stay in sync.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "docs" / "examples"
PYTHON = sys.executable


def _add_xla_flags(env, flags):
    existing = env.get("XLA_FLAGS", "")
    missing = [flag for flag in flags if flag not in existing]
    env["XLA_FLAGS"] = " ".join([*missing, existing]).strip()


def _run_example(name):
    """Run a docs/examples/ script and assert it exits cleanly."""
    script = EXAMPLES_DIR / name
    assert script.exists(), f"Example script not found: {script}"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT / "src")
    )
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_DISABLE_JIT"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    _add_xla_flags(
        env,
        [
            "--xla_cpu_multi_thread_eigen=false",
            "intra_op_parallelism_threads=1",
            "--xla_force_host_platform_device_count=1",
        ],
    )
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{name} failed (exit code {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def test_unconditional_distribution():
    _run_example("unconditional_distribution.py")


def test_conditional_distribution():
    _run_example("conditional_distribution.py")


def test_gp_1d_prior_draws():
    _run_example("gp_1d_prior_draws.py")


def test_gp_2d_prior_draws():
    _run_example("gp_2d_prior_draws.py")


def test_gp_2d_m1sq_prior_draws():
    _run_example("gp_2d_m1sq_prior_draws.py")


def test_multipeak_on_grid():
    _run_example("multipeak_on_grid.py")


def test_build_skymap_from_catalog():
    _run_example("build_skymap_from_catalog.py")


def test_custom_density_skymap():
    _run_example("custom_density_skymap.py")
