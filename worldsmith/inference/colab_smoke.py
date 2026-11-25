"""Run the owner inference smoke test in the existing assigned Colab VM."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from worldsmith.colab.session import (
    ColabError, ColabSession, REMOTE_PROJECT, STATE_DIR, build_environments,
    session_lock, start_session, stop_tracked_session,
)
from worldsmith.colab.storage import R2

from .config import MODEL, MODEL_FILES, MODEL_REVISION, OPENAI_REQUIREMENT, TRANSFORMERS_REQUIREMENT, VLLM_REQUIREMENT


# Keep the SLM environment separate from simulator dependency groups. vLLM's
# pinned numba requires NumPy <2.3; Pymunk 7.3.1 works with NumPy 2.2.6.
SLM_REQUIREMENTS = (
    VLLM_REQUIREMENT, OPENAI_REQUIREMENT, TRANSFORMERS_REQUIREMENT,
    "numpy==2.2.6", "pymunk==7.3.1",
)


def run_session_smoke(*, gpu: str = "L4") -> int:
    status = build_environments("pymunk")
    if status:
        return status
    session = start_session("pymunk", gpu=gpu)
    try:
        smoke_test()
    finally:
        try:
            # Retain diagnostics before releasing the VM, including failed loads.
            logs = STATE_DIR / session.name
            logs.mkdir(parents=True, exist_ok=True)
            for filename in ("slm-install.log", "slm-smoke.log", "vllm.log"):
                try:
                    session._command("download", f"{REMOTE_PROJECT}/{filename}", str(logs / filename))
                except (ColabError, OSError) as error:
                    # Early installation failures may not have created all logs.
                    print(f"Could not retrieve {filename}: {error}", file=sys.stderr)
            print(f"Smoke logs: {logs}")
        finally:
            stop_tracked_session(expected_name=session.name)
    return 0


def smoke_test():
    # Share lifecycle/evaluation locks: inference reserves GPU memory and must
    # finish before owner shutdown or another evaluation starts.
    with session_lock("lifecycle"), session_lock("evaluation"):
        session = ColabSession.active()
        storage = R2()
        files = {}
        for name in MODEL_FILES:
            key = f"worldsmith/cache/models/{MODEL}/{MODEL_REVISION}/{name}"
            cached = storage.head(key)
            if cached is not None:
                files[name] = {"get": storage.signed_url(key, expires=3600), "etag": cached["ETag"].strip('"')}
            else:
                files[name] = {"put": storage.signed_url(key, expires=3600, upload=True), "headers": {}}
        # Upload signed URLs as data, never as code recorded in CLI history.
        with TemporaryDirectory() as temporary:
            plan = Path(temporary) / "model-cache.json"
            plan.write_text(json.dumps(files))
            session._command("upload", str(plan), f"{REMOTE_PROJECT}/model-cache.json")
        pins = dict(requirement.split("==") for requirement in SLM_REQUIREMENTS)
        check_versions = dedent(f"""\
            import sys
            from importlib.metadata import PackageNotFoundError, version

            try:
                ready = all(version(name) == expected for name, expected in {pins!r}.items())
                version('worldsmith')
            except PackageNotFoundError:
                ready = False
            sys.exit(0 if ready else 1)
        """)
        code = dedent(f"""\
            import os
            import subprocess
            from pathlib import Path

            root = Path({REMOTE_PROJECT!r})
            tool = sorted((root / 'tools').iterdir())[0]
            uv = str(tool / 'uv')
            python = str(root / '.slm/bin/python')
            model_cache = root / 'hf-cache'
            with (root / 'slm-install.log').open('w') as log:
                if not Path(python).is_file():
                    subprocess.run(
                        [uv, 'venv', str(root / '.slm'), '--python', str(tool / 'python/bin/python3')],
                        check=True, stdout=log, stderr=subprocess.STDOUT,
                    )
                versions = subprocess.run([python, '-c', {check_versions!r}], stdout=log, stderr=subprocess.STDOUT)
                if versions.returncode:
                    subprocess.run(
                        [uv, 'pip', 'install', '--python', python,
                         *{SLM_REQUIREMENTS!r},
                         str(next((root / 'packages').glob('*.whl')))],
                        check=True, stdout=log, stderr=subprocess.STDOUT,
                    )
                subprocess.run(
                    [python, '-m', 'worldsmith.inference.model_cache', str(root / 'model-cache.json')],
                    check=True, stdout=log, stderr=subprocess.STDOUT,
                )
            with (root / 'slm-smoke.log').open('w') as log:
                subprocess.run(
                    [python, '-m', 'worldsmith.inference.robot_smoke'], cwd=root,
                    env={{**os.environ, 'HF_HUB_CACHE': str(model_cache), 'HF_HUB_OFFLINE': '1'}},
                    check=True, stdout=log, stderr=subprocess.STDOUT,
                )
            print((root / 'slm-smoke.log').read_text())
            print('WORLDSMITH_SLM_SMOKE_PASSED')
        """)
        output = session.execute(code, timeout=3600)
        if "WORLDSMITH_SLM_SMOKE_PASSED" not in output.splitlines():
            raise ColabError(
                "SLM smoke test failed. Inspect /content/worldsmith/"
                "{slm-install.log,slm-smoke.log,vllm.log} in the existing Colab session",
            )
        print(output, end="")
