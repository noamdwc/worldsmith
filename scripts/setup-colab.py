"""Install and smoke-test the evaluation environments inside the Colab VM."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "simulation_environments"


def stage(name, *command, cwd=ROOT):
    print(f"WORLDSMITH_SETUP {name}", flush=True)
    result = subprocess.run(
        command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError(f"{name} exited with status {result.returncode}")


def main():
    print("WORLDSMITH_SETUP NVIDIA L4 verification", flush=True)
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if not gpus or any(gpu not in {"NVIDIA L4", "Tesla L4"} for gpu in gpus):
        print("WORLDSMITH_SETUP GPU mismatch (expected NVIDIA L4)", flush=True)
        raise RuntimeError(f"Expected NVIDIA L4, got {gpus!r}")

    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    stage("Linux system dependencies", "apt-get", "update", "-qq")
    stage(
        "Linux graphics dependencies", "apt-get", "install", "-y", "-qq",
        "libvulkan1", "libegl1", "libgl1", "libxext6", "libglib2.0-0",
    )
    stage("uv installation", sys.executable, "-m", "pip", "install", "-q", "uv")
    stage("Pymunk installation", "uv", "sync", "--locked", "--no-dev", "--python", "3.12")
    for group, version in (("maniskill", "3.11"), ("robosuite", "3.12")):
        directory = ENVIRONMENTS / group
        stage(f"{group} environment", "uv", "venv", "--python", version, cwd=directory)
        # The native simulator locks are macOS-only. Use their direct pins on Linux.
        stage(
            f"{group} installation", "uv", "pip", "install", "--python", ".venv/bin/python",
            "-r", "pyproject.toml", cwd=directory,
        )
    stage(
        "ManiSkill Go2 assets", str(ENVIRONMENTS / "maniskill/.venv/bin/python"),
        "-m", "mani_skill.utils.download_asset", "unitree_go2", "-y",
    )
    stage("LIBERO-PRO installation and assets", "sh", str(ENVIRONMENTS / "libero/setup.sh"))
    with TemporaryDirectory() as temporary:
        for group, task in (
            ("pymunk", "navigation"),
            ("maniskill", "quadruped_reach"),
            ("libero", "open_middle_drawer"),
            ("robosuite", "open_latched_door"),
        ):
            environment = ROOT if group == "pymunk" else ENVIRONMENTS / group
            stage(
                f"{group} smoke episode", str(environment / ".venv/bin/python"),
                "-B", "-m", "worldsmith.benchmark.evaluator", "--task", task, "--episodes", "1",
                "--output", str(Path(temporary) / group),
            )
    print("WORLDSMITH_COLAB_READY", flush=True)


if __name__ == "__main__":
    main()
