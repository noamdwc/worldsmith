"""Prepare simulator caches in a temporary VM or install them for research."""

import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from hashlib import file_digest, md5
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "simulation_environments"
LIBERO = ENVIRONMENTS / "libero"
SMOKE_TASKS = {
    "pymunk": "interception", "maniskill": "quadruped_reach",
    "libero": "open_middle_drawer", "robosuite": "open_latched_door",
}


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


def cache_digest(path):
    with path.open("rb") as file:
        return file_digest(file, lambda: md5(usedforsecurity=False))


def publish(cache, bundle):
    digest = cache_digest(bundle)
    headers = {
        **cache["headers"],
        "Content-Length": str(bundle.stat().st_size),
        "Content-MD5": base64.b64encode(digest.digest()).decode("ascii"),
        "If-None-Match": "*",
    }
    try:
        with bundle.open("rb") as body, urlopen(
            Request(cache["put"], data=body, method="PUT", headers=headers), timeout=120,
        ) as response:
            if response.headers.get("ETag", "").strip('"') != digest.hexdigest():
                raise RuntimeError(f"R2 checksum confirmation failed for {bundle.name}")
    except HTTPError as error:
        if error.code != 412:
            raise RuntimeError(f"R2 upload failed for {bundle.name} (HTTP {error.code})") from None
        # The owner checks the committed cache metadata after a concurrent write.
    except (OSError, HTTPException):
        raise RuntimeError(f"R2 upload failed for {bundle.name}") from None


def unpack(bundle, destination):
    with tarfile.open(bundle) as archive:
        archive.extractall(destination, filter="data")


def restore(cache, bundle, destination):
    download(cache["get"], bundle)
    if cache_digest(bundle).hexdigest() != cache["etag"]:
        raise RuntimeError(f"R2 checksum mismatch for {bundle.name}")
    unpack(bundle, destination)


def archive_directory(directory, bundle, *, gzip=True):
    with tarfile.open(bundle, "w:gz" if gzip else "w") as archive:
        for entry in sorted(directory.iterdir()):
            if entry.name != ".git":
                archive.add(entry, arcname=entry.name)


def install_tools(config, staging, *, prepare):
    tools = {}
    if prepare and any("put" in cache for cache in config["tools"].values()):
        downloads = staging / "bootstrap-uv"
        stage(
            "uv wheel download", sys.executable, "-m", "pip", "download",
            "--dest", str(downloads), "--no-deps",
            "--only-binary", ":all:", f"uv=={config['uv_version']}",
        )
        [wheel] = downloads.glob("uv-*.whl")
        builder = downloads / "uv"
        # The wheel specifies the executable. pip's script install path depends
        # on the host Python's installation scheme.
        with ZipFile(wheel) as archive:
            [executable] = [name for name in archive.namelist() if name.endswith(".data/scripts/uv")]
            with archive.open(executable) as source, builder.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        builder.chmod(0o755)
    for version, cache in config["tools"].items():
        directory = ROOT / "tools" / version
        directory.mkdir(parents=True)
        bundle = staging / f"tools-{version}.tar.gz"
        print(f"WORLDSMITH_SETUP Python {version} tools", flush=True)
        if prepare and "put" in cache:
            managed = staging / f"managed-{version}"
            stage(
                f"Python {version} download", str(builder), "python", "install", version,
                "--install-dir", str(managed), "--no-bin",
            )
            [distribution] = managed.glob(f"cpython-{version}-*")
            shutil.move(distribution, directory / "python")
            shutil.copy2(builder, directory / "uv")
            archive_directory(directory, bundle)
        else:
            restore(cache, bundle, directory)
        uv = str(directory / "uv")
        python = str(directory / "python/bin/python3")
        stage(f"Python {version} verification", python, "--version")
        stage(f"uv {version} verification", uv, "--version")
        if prepare and "put" in cache:
            publish(cache, bundle)
        bundle.unlink()
        tools[version] = (uv, python)
    return tools


def install_environment(group, cache, tools, staging, *, prepare):
    uv, interpreter = tools[cache["python"]]
    environment = ROOT if group == "pymunk" else ENVIRONMENTS / group
    venv = environment / ".venv"
    python = str(venv / "bin/python")
    stage(f"{group} environment", uv, "--offline", "venv", str(venv), "--python", interpreter)
    pins = staging / f"{group}-requirements.txt"
    selection = ("--only-group", "simulation") if group == "pymunk" else ()
    stage(
        f"{group} locked requirements export",
        uv, "--offline", "export", "--frozen", "--project", str(environment),
        "--no-default-groups", "--no-emit-project", "--no-hashes",
        "--format", "requirements-txt", "--output-file", str(pins), *selection,
    )
    wheelhouse = staging / f"{group}-wheels"
    bundle = staging / f"{group}-wheels.tar"
    wheelhouse.mkdir()
    print(f"WORLDSMITH_SETUP {group} wheel cache", flush=True)
    if prepare and "put" in cache:
        # Every runtime dependency is exported from uv.lock. Build those exact
        # versions for the target interpreter, then install the resulting archive.
        stage(
            f"{group} wheel build", sys.executable, "-m", "pip", "--python", python,
            "wheel", "--no-deps", "--wheel-dir", str(wheelhouse),
            "--extra-index-url", "https://download.pytorch.org/whl/cpu", "-r", str(pins),
        )
        archive_directory(wheelhouse, bundle, gzip=False)
        shutil.rmtree(wheelhouse)
        unpack(bundle, wheelhouse)
    else:
        restore(cache, bundle, wheelhouse)
    stage(
        f"{group} offline dependency installation",
        uv, "--offline", "pip", "install", "--python", python, "--no-index", "--no-deps",
        "--find-links", str(wheelhouse), "--only-binary", ":all:", "-r", str(pins),
    )
    if prepare:
        stage(f"{group} dependency verification", uv, "--offline", "pip", "check", "--python", python)
        if "put" in cache:
            print(f"WORLDSMITH_SETUP {group} wheel cache upload", flush=True)
            publish(cache, bundle)
    shutil.rmtree(wheelhouse)
    bundle.unlink()
    return python


def verify_go2(directory):
    urdf_directory = directory / "robots/unitree_go2/urdf"
    for name in ("go2_description.urdf", "go2_description_simplified_locomotion.urdf"):
        urdf = ElementTree.parse(urdf_directory / name)
        for mesh in urdf.iterfind(".//mesh"):
            if not (urdf_directory / mesh.attrib["filename"]).is_file():
                raise RuntimeError(f"Missing Go2 mesh in {name}")


def install_assets(assets, pythons, staging, *, prepare):
    for group, cache in assets.items():
        bundle = staging / f"{group}-assets.tar.gz"
        print(f"WORLDSMITH_SETUP {group} asset cache", flush=True)
        if group == "maniskill":
            directory = ROOT / "maniskill-assets"
            directory.mkdir()
            os.environ["MS_ASSET_DIR"] = str(directory)
            if prepare and "put" in cache:
                stage(
                    "ManiSkill Go2 asset download", pythons[group], "-m",
                    "mani_skill.utils.download_asset", "unitree_go2", "-y",
                )
            else:
                restore(cache, bundle, directory)
            # ManiSkill adds a data/ directory under MS_ASSET_DIR.
            verify_go2(directory / "data")
        else:
            directory = LIBERO / "source"
            os.environ["WORLDSMITH_LIBERO_INSTALL"] = str(LIBERO)
            if prepare and "put" in cache:
                stage("LIBERO-PRO asset download", "sh", str(LIBERO / "setup.sh"), "--assets-only")
            else:
                restore(cache, bundle, directory)
            stage("LIBERO-PRO asset verification and configuration", "sh", str(LIBERO / "setup.sh"), "--configure")

        if prepare and "put" in cache:
            archive_directory(directory, bundle)
            publish(cache, bundle)
        bundle.unlink()


def main():
    config_path = ROOT / "setup.json"
    config = json.loads(config_path.read_text())
    config_path.unlink()

    print("WORLDSMITH_SETUP Linux x86_64 verification", flush=True)
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("Worldsmith requires a Linux x86-64 Colab VM")

    print("WORLDSMITH_SETUP NVIDIA L4 verification", flush=True)
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).strip().splitlines()
    if not gpus or any(gpu not in {"NVIDIA L4", "Tesla L4"} for gpu in gpus):
        print("WORLDSMITH_SETUP GPU mismatch (expected NVIDIA L4)", flush=True)
        raise RuntimeError(f"Expected NVIDIA L4, got {gpus!r}")

    prepare = config["mode"] == "prepare"
    if prepare and any("put" in cache for cache in config["wheelhouses"].values()):
        os.environ["DEBIAN_FRONTEND"] = "noninteractive"
        stage("Linux build package index", "apt-get", "update", "-qq")
        stage(
            "Linux wheel build dependencies", "apt-get", "install", "-y", "-qq",
            "build-essential", "linux-libc-dev",
        )

    with TemporaryDirectory() as temporary:
        staging = Path(temporary)
        tools = install_tools(config, staging, prepare=prepare)
        pythons = {
            group: install_environment(group, cache, tools, staging, prepare=prepare)
            for group, cache in config["wheelhouses"].items()
        }
        install_assets(config["assets"], pythons, staging, prepare=prepare)
        if not prepare:
            wheel = staging / config["wheel"]["filename"]
            print("WORLDSMITH_SETUP Worldsmith wheel download", flush=True)
            download(config["wheel"]["url"], wheel)
            for group, python in pythons.items():
                uv, _ = tools[config["wheelhouses"][group]["python"]]
                stage(
                    f"{group} Worldsmith installation",
                    uv, "--offline", "pip", "install", "--python", python, "--no-index", "--no-deps", str(wheel),
                )
                output = staging / group
                stage(
                    f"{group} smoke episode", python, "-B", "-m", "worldsmith.benchmark.evaluator",
                    "--task", SMOKE_TASKS[group], "--episodes", "1", "--output", str(output),
                )
                if not (output / "summary.json").is_file():
                    raise RuntimeError(f"{group} smoke episode produced no result")
    print("WORLDSMITH_COLAB_PREPARED" if prepare else "WORLDSMITH_COLAB_READY", flush=True)


if __name__ == "__main__":
    main()
