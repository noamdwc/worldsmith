import json
import subprocess
import sys
from io import StringIO

import pytest

from worldsmith.api import Action
from worldsmith.benchmark.evaluate import evaluate
from worldsmith.benchmark.simulation import TIME_STEP
from worldsmith.benchmark.tasks.interception import SPLITS, TARGET_NOISE, Chase, Interception, Lead


def test_seed_replays_noisy_observations():
    world = Interception().start(12)
    replay = Interception().start(12)
    observation = world.initial_observation

    assert observation == replay.initial_observation
    assert observation.entities[observation.robot_id].position != (0.0, 0.0)
    assert observation.entities["target"].covariance == ((TARGET_NOISE**2, 0.0), (0.0, TARGET_NOISE**2))
    assert world.step(Action(0.0)) == replay.step(Action(0.0))
    assert world.step(Action(0.0)).time == 2 * TIME_STEP


@pytest.mark.parametrize("action", [Action(float("nan")), Action(0.0, 1.1)])
def test_invalid_action_is_rejected(action):
    with pytest.raises(ValueError, match="finite heading and throttle"):
        Interception().start(12).step(action)


def test_leading_improves_development_capture_rate():
    seeds = list(SPLITS["dev"])
    chase = evaluate(Interception(), Chase, seeds, StringIO())
    lead = evaluate(Interception(), Lead, seeds, StringIO())
    assert lead["captures"] > chase["captures"]


def test_cli_loads_factory_and_writes_run_files(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable, "-m", "worldsmith", "evaluate",
            "--controller", "worldsmith.benchmark.tasks.interception:Lead",
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
