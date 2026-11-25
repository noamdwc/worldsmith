from functools import partial
from importlib.util import find_spec
from inspect import signature
from pathlib import Path

from worldsmith.benchmark.evaluator import evaluate_seeds_to_directory
from worldsmith.benchmark.runner import ControllerError
from worldsmith.benchmark.task_catalog import TASK_CATALOG


def evaluate_tasks(task_seeds: dict[str, list[int]], output: Path, *, workers: int) -> None:
    output = output.resolve()
    output.mkdir(parents=True)
    try:
        try:
            # The working directory holds either the live workspace or a saved snapshot.
            controller_path = Path.cwd() / "autoresearch/controller.py"
            if not controller_path.is_file():
                raise FileNotFoundError(f"Controller file is missing: {controller_path}")
            spec = find_spec("autoresearch.controller")
            if spec is None or spec.origin is None or Path(spec.origin).resolve() != controller_path:
                raise ImportError(f"Controller import does not resolve to the expected file: {controller_path}")

            from autoresearch.controller import make_controller

            parameters = signature(make_controller).parameters
        except Exception as exc:
            raise ControllerError(
                f"Controller import: {type(exc).__name__}: {exc}",
                public_message="Could not load controller.py or its make_controller factory",
            ) from exc

        for task, seeds in task_seeds.items():
            (output / "task.txt").write_text(task, encoding="utf-8")
            open_adapter = TASK_CATALOG[task]
            factory = partial(make_controller, task=task) if "task" in parameters else make_controller
            evaluate_seeds_to_directory(
                open_adapter, factory, seeds, output / task, task=task,
                workers=workers,
            )
    except ControllerError as exc:
        (output / "controller_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
        (output / "controller_feedback.txt").write_text(exc.public_message + "\n", encoding="utf-8")
        raise
