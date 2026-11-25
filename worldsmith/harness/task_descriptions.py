"""Task goals for simulators whose adapter metadata does not supply an objective."""

OBJECTIVES = {
    "push_cube": "Move the cube center within 0.1 m of the goal in XY, at table height.",
    "pick_cube": "Move the cube center within 0.025 m of the 3D goal, with the arm still.",
    "stack_cube": "Place cube A on cube B, still and released.",
    "peg_insertion_side": "Insert the peg into the hole. In the hole frame the peg head must reach X >= -0.015 m with Y and Z within the hole radius.",
    "pull_cube": "Move the cube center within 0.1 m of the goal in XY.",
    "push_t": "Make the T cover at least 90% of its goal outline.",
    "roll_ball": "Move the ball center within 0.1 m of the goal in XY.",
    "quadruped_reach": "Reach within 0.35 m of the ground-plane goal without falling.",
    "open_middle_drawer": "Open the middle drawer by more than 0.14 m along its slide (negative drawer_slide indicates opening).",
    "place_bowl_on_plate": "Place the designated black bowl between the plate and ramekin onto the plate.",
    "place_two_items_in_basket": "Put both the alphabet soup and tomato sauce inside the basket.",
    "open_latched_door": "Open the latched door past 0.3 radians.",
    "assemble_nuts": "Seat both nuts on their matching pegs and move the gripper away.",
    "handover_hammer": "The receiving arm must grasp the handle while the pickup arm releases the hammer, with the hammer's lower end more than 0.1 m above the table.",
    "wipe_table": "Clear all dirt marks with the passive wiping tool, avoiding arm collisions and joint limits.",
}
