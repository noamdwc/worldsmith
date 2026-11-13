"""Headless native arm tasks with structured observations and native success."""

import os
from contextlib import contextmanager
from functools import partial
from importlib.metadata import version

import numpy as np

from worldsmith.benchmark.runner import InvalidAction

TASKS = {
    "open_latched_door": ("Door", 1, 300),
    "assemble_nuts": ("NutAssembly", 1, 450),
    "handover_hammer": ("TwoArmHandover", 2, 400),
}
ARM_FIELDS = ("joint_pos", "joint_vel", "eef_pos", "eef_quat_site", "gripper_qpos", "gripper_qvel")


class RobosuiteAdapter:
    def __init__(self, task, make_env):
        self.task = task
        self.make_env = make_env
        self.native_task, self.arms, self.horizon = TASKS[task]
        self.env = None

    def observation(self, native):
        observation = {
            f"robot{arm}_{field}": native[f"robot{arm}_{field}"]
            for arm in range(self.arms) for field in ARM_FIELDS
        }
        env = self.env
        data = env.sim.data
        if self.native_task == "Door":
            hinge = env.sim.model.joint_name2id(env.door.joints[0])
            observation.update(
                handle_pos=native["handle_pos"],
                hinge_pos=data.xanchor[hinge].copy(),
                hinge_angle=float(native["hinge_qpos"]),
                latch_angle=float(native["handle_qpos"]),
            )
        elif self.native_task == "NutAssembly":
            for nut, name, peg in zip(
                env.nuts, ("square", "round"), (env.peg1_body_id, env.peg2_body_id), strict=True,
            ):
                observation[f"{name}_nut_pos"] = native[f"{nut.name}_pos"]
                observation[f"{name}_nut_quat"] = native[f"{nut.name}_quat"]
                observation[f"{name}_nut_handle_pos"] = data.get_site_xpos(nut.important_sites["handle"]).copy()
                observation[f"{name}_peg_pos"] = data.body_xpos[peg].copy()
        else:
            observation.update(
                hammer_pos=native["hammer_pos"],
                hammer_quat=native["hammer_quat"],
                hammer_handle_pos=native["handle_xpos"],
                table_height=float(data.site_xpos[env.table_top_id][2]),
            )
        return observation

    def reset(self, seed):
        self.close()
        # The handover environment samples hammer geometry when constructing the model.
        # Build once per seed so geometry and placement are independent of episode order.
        self.env = self.make_env(seed=seed)
        self.low, self.high = self.env.action_spec
        return self.observation(self.env.reset()), {"native_seed": seed}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def encode_action(self, action):
        if (
            not isinstance(action, np.ndarray)
            or action.shape != self.low.shape
            or action.dtype.kind != "f"
            or not ((self.low <= action) & (action <= self.high)).all()
        ):
            raise InvalidAction(
                f"Expected a flat array of {self.low.size} finite floating-point commands in [-1, 1]"
            )
        return action.tolist()

    def step(self, action):
        native, reward, at_limit, info = self.env.step(action)
        success = bool(self.env._check_success())
        return self.observation(native), float(reward), success, at_limit and not success, {
            **info, "success": success,
        }

    def score(self, episode):
        return {"success": episode["steps"][-1]["info"]["success"]}

    def summarize(self, episodes):
        return {"success_rate": sum(row["success"] for row in episodes) / len(episodes)}

    def metadata(self):
        return {
            "task_id": self.task,
            "native_task": self.native_task,
            "robots": ["Panda"] * self.arms,
            "environment_configuration": self.env.env_configuration,
            "max_episode_steps": self.horizon,
            "control_frequency": self.env.control_freq,
            "action_shape": list(self.low.shape),
            "action_mode": "OSC_POSE, world-frame delta",
            "observation_mode": "structured, no cameras",
            "reward_shaping": self.env.reward_shaping,
            "versions": {name: version(name) for name in ("robosuite", "mujoco", "numpy")},
        }


@contextmanager
def open_robosuite(task):
    # MuJoCo reads this on import. No render contexts are created during evaluation.
    os.environ["MUJOCO_GL"] = "disable"
    from robosuite import make
    from robosuite.controllers import load_composite_controller_config

    native_task, arms, horizon = TASKS[task]
    controller = load_composite_controller_config(robot="Panda")
    controller["body_parts"]["right"]["input_ref_frame"] = "world"
    make_env = partial(
        make, native_task, robots=["Panda"] * arms, controller_configs=controller,
        use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
        hard_reset=False, horizon=horizon, reward_shaping=False,
    )
    adapter = RobosuiteAdapter(task, make_env)
    try:
        yield adapter
    finally:
        adapter.close()


def evaluation_setup(task):
    _, arms, _ = TASKS[task]

    def idle(observation):
        return np.tile([0., 0., 0., 0., 0., 0., -1.], arms)

    return partial(open_robosuite, task), lambda: idle
