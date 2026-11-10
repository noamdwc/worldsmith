import json
import subprocess
import sys
from io import StringIO

import pytest

from worldsmith.api import Action
from worldsmith.benchmark.baselines import Chase, Lead
from worldsmith.benchmark.evaluate import SPLITS, evaluate
from worldsmith.benchmark.interception import TARGET_NOISE, TIME_STEP, Interception


def test_seed_replays_noisy_observations():
    world = Interception(12)
    replay = Interception(12)
    observation = world.initial_observation

    assert observation == replay.initial_observation
    assert observation.robot.position != (0.0, 0.0)
    assert observation.target.covariance == ((TARGET_NOISE**2, 0.0), (0.0, TARGET_NOISE**2))
    assert world.step(Action(0.0)) == replay.step(Action(0.0))
    assert world.step(Action(0.0)).time == 2 * TIME_STEP


@pytest.mark.parametrize("action", [Action(float("nan")), Action(0.0, 1.1)])
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError, match="finite heading and throttle"):
        Interception(12).step(action)


def test_leading_improves_development_capture_rate():
    seeds = list(SPLITS["dev"])
    chase = evaluate(Chase, seeds, StringIO())
    lead = evaluate(Lead, seeds, StringIO())
    assert lead["captures"] > chase["captures"]


def test_cli_loads_factory_and_writes_run_files(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable, "-m", "worldsmith", "evaluate",
            "--controller", "worldsmith.benchmark.baselines:Lead",
            "--seed", "12", "--output", str(output),
        ],
        check=True, capture_output=True, text=True,
    )

    assert result.stdout == (output / "summary.json").read_text()
    summary = json.loads(result.stdout)
    assert summary["seeds"] == [12]
    episode = json.loads((output / "episodes.jsonl").read_text())
    assert episode["seed"] == 12
    assert episode["steps"] == len(episode["frames"])
    assert summary["captures"] == int(episode["captured"])
