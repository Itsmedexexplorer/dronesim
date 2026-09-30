# Navigation and semantic mapping changes

- Nav2: A*, known-space planning, 10 Hz local/3 Hz global costmaps,
  0.15 m displacement within 4 seconds required for progress.
- Explorer: three consecutive scan obstructions within the next metre of
  the published route trigger cancellation. Failed goals have a 20-second,
  1-metre-radius cooldown, including inspection and fallback dispatches.
  Goal generations prevent late action results from completing a newer goal.
  Progress uses displacement rather than decreasing straight-line goal distance.
- Confirmed objects retain their IDs and positions on reobservation; confidence
  and last-seen time update. Association checks size compatibility and proximity;
  a limited uncertainty/overlap gate extends matching to at most 0.85 m.
  Ambiguous/conflicting nearby detections are deferred rather than registered.
- The map consumes survivor marker IDs, not rounded coordinates. LiDAR obstacles
  are retained at survivor positions and the entrance; no forced-free gate mask.

No survivor coordinates are introduced by these changes. Existing mission
waypoints still assume the supplied arena layout. Restart the mission/map when
changing the simulation layout; moving objects during a mission is not a tested
dynamic-object tracking use case.

## Checks

From the project root:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m unittest -v test_perception_safety.py
python3 test_tri_modal_pipeline.py
```

Eleven regression tests and the synthetic classifier/depth/relocation checks
passed. They are not an autonomous Gazebo acceptance test.

## Remaining validation and limitations

Run full missions with randomized mannequin positions, blocked corridors, and
nearby pairs. Measure unique detections, false positives, position error, time
to coverage, and cancellation-to-new-route latency. Ground truth must be used
only by the evaluator, never fed to the detector.

The mapper is a pose-projected occupancy grid, not a loop-closing SLAM backend;
these changes do not fix pose drift or add scan matching. Cooldown is goal-level;
Nav2's obstacle costmap handles blocked cells, not a new corridor graph blacklist.
The simulation appearance heuristic is not a trained human/mannequin detector.
The association variance is a tuning approximation, not calibrated uncertainty.
Close people may remain ambiguous. Complete visual coverage, timestamp-aligned
pose fusion, and reliable real-human detection need separate validation/work.
