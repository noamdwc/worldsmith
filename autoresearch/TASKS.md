# Public three-task benchmark

One shared `make_controller()` handles all three goals. The researcher can read these task descriptions and all development traces. V1 uses no runtime LLM and does not require task blindness.

All tasks use an open 2D plane and 0.1-second steps. The [World API](API.md) supplies noisy positions and measurement covariances, observation time, robot ID, maximum speed (2.4 units/second), and the goal identifier. Entity velocity is unobserved. Moving entities have constant velocity within each episode.

## Moving-target interception

`observation.goal == "interception"`. Get within 0.3 units of `target` within 45 steps. Touching the capture region once is enough; matching the target's motion is unnecessary.

Metrics: success rate (also called capture rate), captures, and mean closest distance. Traces include per-step true distance.

## Moving-obstacle navigation

`observation.goal == "navigation"`. Reach the stationary `destination` within 0.35 units in 80 steps while avoiding `obstacle_0` and `obstacle_1`, which cross the route in opposite directions. Robot and obstacle radii are 0.15 and 0.55 units. A center distance of 0.7 units or less counts as a collision and makes the episode unsuccessful, even if the robot later reaches the destination.

Contact is scored at the initial state and each simulation step; bodies do not bounce or block each other. There are no walls. Predicting crossing traffic, choosing a route, and waiting can help.

Destinations are 8–10 units ahead with a small lateral offset. Obstacle crossing speeds vary from 0.9–1.7 and 0.8–1.6 units/second, with independently sampled starting positions. Turning away only when an obstacle is nearby can be too late for faster crossings.

Metrics: success rate, collision rate, mean final destination distance, and mean minimum obstacle clearance (center distance minus 0.7). Traces include destination distance and obstacle clearance at each step.

## Rendezvous / tracking

`observation.goal == "rendezvous"`. Approach `target` and remain within 0.6 units for 20 consecutive control steps (2 seconds) within a 100-step episode. Leaving the radius resets the consecutive-step count. The initial observation does not count as a control step. Success requires sustained proximity, so controlling speed and matching motion matter.

Targets start 3–5 units away and move at 0.8–1.7 units/second in a sampled direction. The robot is always faster, with enough time to catch even an outbound target and sustain proximity. Slowing down based only on current distance can leave a persistent tracking lag; estimating motion can help match speed while remaining close.

Metrics: success rate, mean longest tracking duration, mean distance over control steps, and mean final distance. Traces include per-step true distance.

## Evaluation and splits

`evaluate_controller()` evaluates all three tasks together on 8 fixed development seeds per task (0–7), for 24 episodes total. Scenarios and noisy measurements are deterministic for a given task, seed, and action sequence. Every episode starts with a fresh controller from the same factory.

The aggregate score is the equally weighted average of the three task success rates. Compare per-task results to see improvements and regressions; a higher aggregate can hide a regression on one task. Secondary metrics are diagnostic and are not normalized or included in the score.

The owner can evaluate held-out seeds (1000–1007). Researchers use development feedback only. Independent interception evaluation preserves its original 32-seed splits (0–31 and 1000–1031), so its full-split results are not directly comparable with the combined run's 8-seed subset.
