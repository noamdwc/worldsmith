"""Structured observations and native scoring for the position-swap tasks."""

import os
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

from worldsmith.benchmark.runner import InvalidAction

INSTALL = Path(__file__).resolve().parents[2] / "libero"
SOURCE = INSTALL / "source"
SOURCE_REVISION = "eafdb809426b13153aa1e4c42d6601844217dfec"
DATA_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
SETTLE_STEPS = 10

TASKS = {
    "open_middle_drawer": ("libero_goal_swap", 300),
    "place_bowl_on_plate": ("libero_spatial_swap", 220),
    "place_two_items_in_basket": ("libero_10_swap", 520),
}


class LiberoAdapter:
    def __init__(self, task, env, states, horizon):
        self.task = task
        self.env = env
        self.states = states
        self.horizon = horizon
        self.low, self.high = env.env.action_spec

    def observation(self, native):
        observation = {
            name: value for name, value in native.items()
            if not name.endswith("-state")
        }
        if self.task == "open_middle_drawer":
            data = self.env.sim.data
            body = "wooden_cabinet_1_cabinet_middle"
            # Center of the handle collision bar in the moving drawer's local frame.
            observation["drawer_handle_pos"] = (
                data.get_body_xpos(body)
                + data.get_body_xmat(body) @ [-0.00278, -0.10161, 0.11041]
            )
            observation["drawer_slide"] = data.get_joint_qpos("wooden_cabinet_1_middle_level")
        return observation

    def reset(self, seed):
        # The seed is the saved-layout index. Wrapping would leak development layouts.
        if not 0 <= seed < len(self.states):
            raise ValueError(f"Use a saved layout index in [0, {len(self.states)})")
        self.env.seed(seed)
        self.env.reset()
        self.env.set_init_state(self.states[seed])
        hold = idle(None)
        for _ in range(SETTLE_STEPS):
            native, _, _, _ = self.env.step(hold)
        self.steps = 0
        return self.observation(native), {"init_state_index": seed}

    def encode_action(self, action):
        import numpy as np

        if (
            not isinstance(action, np.ndarray)
            or action.shape != self.low.shape
            or action.dtype.kind != "f"
            or not ((self.low <= action) & (action <= self.high)).all()
        ):
            raise InvalidAction("Expected seven finite floating-point commands in [-1, 1]")
        return action.tolist()

    def step(self, action):
        native, reward, success, info = self.env.step(action)
        self.steps += 1
        success = bool(success)
        return (
            self.observation(native), float(reward), success,
            self.steps == self.horizon and not success, {**info, "success": success},
        )

    def score(self, episode):
        return {"success": episode["steps"][-1]["info"]["success"]}

    def summarize(self, episodes):
        return {"success_rate": sum(row["success"] for row in episodes) / len(episodes)}

    def metadata(self):
        return {
            "task_id": self.task,
            "native_suite": TASKS[self.task][0],
            "native_task_index": 0,
            "source_revision": SOURCE_REVISION,
            "dataset_revision": DATA_REVISION,
            "initial_state_count": len(self.states),
            "settle_steps": SETTLE_STEPS,
            "max_episode_steps": self.horizon,
            "action_shape": list(self.low.shape),
            "action_mode": "OSC_POSE",
            "observation_mode": "structured, no cameras",
            "versions": {name: version(name) for name in ("mujoco", "robosuite", "numpy", "torch")},
        }


@contextmanager
def open_libero(task):
    if not (INSTALL / "config" / "config.yaml").is_file():
        raise RuntimeError("Run libero/setup.sh before evaluating this benchmark")
    # Set before the first native import, including for direct evaluator invocations.
    os.environ["MUJOCO_GL"] = "disable"
    os.environ["LIBERO_CONFIG_PATH"] = str(INSTALL / "config")
    os.environ.setdefault("MPLCONFIGDIR", str(INSTALL / "config"))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    import torch

    suite, horizon = TASKS[task]
    benchmark = get_benchmark(suite)()
    definition = benchmark.get_task(0)
    bddl = Path(benchmark.get_task_bddl_file_path(0))
    state_file = Path(get_libero_path("init_states")) / suite / definition.init_states_file
    hashes = {}
    for line in (INSTALL / "assets.sha256").read_text().splitlines():
        digest, path = line.split()
        hashes[path] = digest
    state_bytes = state_file.read_bytes()
    for artifact, contents in ((bddl, bddl.read_bytes()), (state_file, state_bytes)):
        relative_path = artifact.relative_to(SOURCE / "libero/libero")
        if sha256(contents).hexdigest() != hashes[str(relative_path)]:
            raise ValueError(f"Benchmark asset differs from the pinned dataset: {artifact.name}")

    # Upstream saves NumPy arrays in legacy PyTorch files. Only load verified bytes.
    states = torch.load(BytesIO(state_bytes), map_location="cpu", weights_only=False)
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    try:
        yield LiberoAdapter(task, env, states, horizon)
    finally:
        env.close()


def idle(observation):
    import numpy as np

    return np.array([0., 0., 0., 0., 0., 0., -1.])


def evaluation_setup(task):
    return partial(open_libero, task), lambda: idle
