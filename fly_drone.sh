#!/bin/bash
# Gentle drone flight - hover at low altitude then fly forward
# Uses the built-in quadcopter.sdf world (world name: quadcopter)

echo "=== Step 1: Unpausing simulation ==="
gz service -s /world/quadcopter/control \
  --reqtype gz.msgs.WorldControl \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'pause: false'

sleep 1

echo "=== Step 2: Gentle takeoff (hover ~2m) ==="
# 620 RPM = gentle lift for the X3 UAV
for i in $(seq 1 60); do
    gz topic -t /X3/gazebo/command/motor_speed \
      --msgtype gz.msgs.Actuators \
      -p 'velocity:[620, 620, 620, 620]'
    sleep 0.05
done

echo "=== Step 3: Stabilize hover ==="
for i in $(seq 1 40); do
    gz topic -t /X3/gazebo/command/motor_speed \
      --msgtype gz.msgs.Actuators \
      -p 'velocity:[600, 600, 600, 600]'
    sleep 0.05
done

echo "=== Step 4: Fly forward (tilt rotors) ==="
# Front rotors slightly slower = drone tilts and moves forward
for i in $(seq 1 80); do
    gz topic -t /X3/gazebo/command/motor_speed \
      --msgtype gz.msgs.Actuators \
      -p 'velocity:[580, 620, 580, 620]'
    sleep 0.05
done

echo "=== Step 5: Hover again ==="
for i in $(seq 1 40); do
    gz topic -t /X3/gazebo/command/motor_speed \
      --msgtype gz.msgs.Actuators \
      -p 'velocity:[600, 600, 600, 600]'
    sleep 0.05
done

echo "=== Done! Drone should be hovering ==="
