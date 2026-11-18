"""Run the owner inference smoke test in the existing assigned Colab VM."""

from textwrap import dedent

from worldsmith.inference.config import OPENAI_REQUIREMENT, TRANSFORMERS_REQUIREMENT, VLLM_REQUIREMENT

from .session import ColabError, ColabSession, REMOTE_PROJECT, session_lock


def smoke_test():
    # Share lifecycle/evaluation locks: inference reserves GPU memory and must
    # finish before owner shutdown or another evaluation starts.
    with session_lock("lifecycle"), session_lock("evaluation"):
        session = ColabSession.active()
        code = dedent(f"""\
            import subprocess
            from pathlib import Path

            root = Path({REMOTE_PROJECT!r})
            tool = sorted((root / 'tools').iterdir())[0]
            uv = str(tool / 'uv')
            python = str(root / '.slm/bin/python')
            with (root / 'slm-install.log').open('w') as log:
                if not Path(python).is_file():
                    subprocess.run(
                        [uv, 'venv', str(root / '.slm'), '--python', str(tool / 'python/bin/python3')],
                        check=True, stdout=log, stderr=subprocess.STDOUT,
                    )
                subprocess.run(
                    [uv, 'pip', 'install', '--python', python,
                     {VLLM_REQUIREMENT!r}, {OPENAI_REQUIREMENT!r}, {TRANSFORMERS_REQUIREMENT!r},
                     str(next((root / 'packages').glob('*.whl')))],
                    check=True, stdout=log, stderr=subprocess.STDOUT,
                )
            with (root / 'slm-smoke.log').open('w') as log:
                subprocess.run(
                    [python, '-m', 'worldsmith.inference.smoke'], cwd=root,
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
