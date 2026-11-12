from functools import partial

import pytest


@pytest.fixture
def research_metrics():
    from worldsmith.benchmark.evaluator import evaluate
    from worldsmith.benchmark.pymunk_adapter import (
        PymunkAdapter, summarize_research, write_research_episode,
    )

    def run(tasks, factory, seeds, trace, *, combined=False):
        return evaluate(
            {name: PymunkAdapter(task) for name, task in tasks.items()},
            factory, seeds, partial(write_research_episode, trace),
            prepare_controller=PymunkAdapter.make_controller,
            summarize=partial(summarize_research, combined=combined),
        )

    return run
