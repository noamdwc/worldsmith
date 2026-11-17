"""Restore Colab's wheelhouses and simulator assets, then smoke-test the runtime."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "simulation_environments"
REQUIREMENTS = ENVIRONMENTS / "colab"
LIBERO = ENVIRONMENTS / "libero"


def stage(name, *command):
    print(f"WORLDSMITH_SETUP {name}", flush=True)
    result = subprocess.run(
        command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError(f"{name} exited with status {result.returncode}")


def download(url, destination):
    try:
        with urlopen(url, timeout=60) as response, destination.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)
    except (OSError, HTTPException):
        raise RuntimeError(f"R2 download failed for {destination.name}") from None


def upload(url, source):
    try:
        with source.open("rb") as body, urlopen(
            Request(
                url, data=body, method="PUT",
                headers={"Content-Length": str(source.stat().st_size), "If-None-Match": "*"},
            ),
            timeout=120,
        ):
            pass
    except HTTPError as error:
        if error.code != 412:
            raise RuntimeError(f"R2 upload failed for {source.name} (HTTP {error.code})") from None
        # Another session already populated the same immutable cache key.
    except (OSError, HTTPException):
        raise RuntimeError(f"R2 upload failed for {source.name}") from None


def restore(cache, bundle, destination):
    download(cache["get"], bundle)
    with tarfile.open(bundle) as archive:
        archive.extractall(destination, filter="data")


def install_environment(group, cache, wheel, staging):
    environment = ROOT if group == "pymunk" else ENVIRONMENTS / group
    venv = environment / ".venv"
    python = venv / "bin/python"
    stage(f"{group} environment", "uv", "venv", str(venv), "--python", cache["python"])
    pins = REQUIREMENTS / f"{group}.txt"
    wheelhouse = staging / f"{group}-wheels"
    bundle = staging / f"{group}-wheels.tar"
    wheelhouse.mkdir()
    print(f"WORLDSMITH_SETUP {group} wheel cache", flush=True)
    if "get" in cache:
        restore(cache, bundle, wheelhouse)
    else:
        # Use the target interpreter for wheel selection and for any sdist builds.
        stage(
            f"{group} wheel build", sys.executable, "-m", "pip", "--python", str(python),
            "wheel", "--no-deps", "--wheel-dir", str(wheelhouse),
            "--extra-index-url", "https://download.pytorch.org/whl/cpu", "-r", str(pins),
        )
        with tarfile.open(bundle, "w") as archive:
            for path in sorted(wheelhouse.glob("*.whl")):
                archive.add(path, arcname=path.name)

    pip_install = ["uv", "pip", "install", "--python", str(python), "--no-index", "--no-deps"]
    stage(
        f"{group} dependency installation", *pip_install, "--find-links", str(wheelhouse),
        "--only-binary", ":all:", "-r", str(pins),
    )
    stage(f"{group} Worldsmith installation", *pip_install, str(wheel))
    if "put" in cache:
        print(f"WORLDSMITH_SETUP {group} wheel cache upload", flush=True)
        upload(cache["put"], bundle)
    shutil.rmtree(wheelhouse)
    bundle.unlink()
    return python


def install_assets(assets, maniskill_python, staging):
    maniskill = ROOT / "maniskill-assets"
    maniskill.mkdir()
    os.environ["MS_ASSET_DIR"] = str(maniskill)
    os.environ["WORLDSMITH_LIBERO_INSTALL"] = str(LIBERO)
    libero_setup = ["sh", str(LIBERO / "setup.sh")]

    for group, path, command in (
        ("maniskill", maniskill, [str(maniskill_python), "-m", "mani_skill.utils.download_asset", "unitree_go2", "-y"]),
        ("libero", LIBERO / "source", [*libero_setup, "--assets-only"]),
    ):
        cache = assets[group]
        bundle = staging / f"{group}-assets.tar.gz"
        print(f"WORLDSMITH_SETUP {group} asset cache", flush=True)
        if "get" in cache:
            restore(cache, bundle, path)
        else:
            stage(f"{group} asset download", *command)
            with tarfile.open(bundle, "w:gz") as archive:
                for entry in sorted(path.iterdir()):
                    if entry.name != ".git":
                        archive.add(entry, arcname=entry.name)
            upload(cache["put"], bundle)
        bundle.unlink()

    stage("LIBERO-PRO asset verification and configuration", *libero_setup, "--configure")


def main():
    config_path = ROOT / "setup.json"
    config = json.loads(config_path.read_text())
    config_path.unlink()

    print("WORLDSMITH_SETUP Linux x86_64 and NVIDIA L4 verification", flush=True)
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("Worldsmith requires a Linux x86-64 Colab VM")
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if not gpus or any(gpu not in {"NVIDIA L4", "Tesla L4"} for gpu in gpus):
        print("WORLDSMITH_SETUP GPU mismatch (expected NVIDIA L4)", flush=True)
        raise RuntimeError(f"Expected NVIDIA L4, got {gpus!r}")

    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    stage("Linux system dependencies", "apt-get", "update", "-qq")
    stage(
        "Linux graphics dependencies", "apt-get", "install", "-y", "-qq",
        "libvulkan1", "libegl1", "libgl1", "libxext6", "libglib2.0-0",
    )
    if any("put" in cache for cache in config["wheelhouses"].values()):
        stage(
            "Linux wheel build dependencies", "apt-get", "install", "-y", "-qq",
            "build-essential", "linux-libc-dev",
        )
    stage("uv installation", sys.executable, "-m", "pip", "install", "-q", "uv")

    with TemporaryDirectory() as temporary:
        staging = Path(temporary)
        wheel = staging / config["wheel"]["filename"]
        print("WORLDSMITH_SETUP Worldsmith wheel download", flush=True)
        download(config["wheel"]["url"], wheel)
        pythons = {
            group: install_environment(group, cache, wheel, staging)
            for group, cache in config["wheelhouses"].items()
        }
        install_assets(config["assets"], pythons["maniskill"], staging)
        for group, task in (
            ("pymunk", "interception"),
            ("maniskill", "quadruped_reach"),
            ("libero", "open_middle_drawer"),
            ("robosuite", "open_latched_door"),
        ):
            stage(
                f"{group} smoke episode", str(pythons[group]),
                "-B", "-m", "worldsmith.benchmark.evaluator", "--task", task, "--episodes", "1",
                "--output", str(staging / group),
            )
    print("WORLDSMITH_COLAB_READY", flush=True)


if __name__ == "__main__":
    main()
