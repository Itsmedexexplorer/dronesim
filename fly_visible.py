import subprocess
import time
import re
import sys

def send_motors(w0, w1, w2, w3):
    cmd = [
        "gz", "topic", "-t", "/X3/gazebo/command/motor_speed",
        "--msgtype", "gz.msgs.Actuators",
        "-p", f"velocity:[{w0}, {w1}, {w2}, {w3}]"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_state():
    try:
        res = subprocess.run(
            ["gz", "topic", "-t", "/world/quadcopter/pose/info", "-e", "-n", "1"],
            capture_output=True, text=True, timeout=2
        )
        match = re.search(r'name:\s*"x3".*?position\s*\{([^}]*)\}', res.stdout, re.DOTALL)
        if match:
            pos_text = match.group(1)
            x_m = re.search(r'x:\s*([0-9\.\-eE]+)', pos_text)
            y_m = re.search(r'y:\s*([0-9\.\-eE]+)', pos_text)
            z_m = re.search(r'z:\s*([0-9\.\-eE]+)', pos_text)
            x = float(x_m.group(1)) if x_m else 0.0
            y = float(y_m.group(1)) if y_m else 0.0
            z = float(z_m.group(1)) if z_m else 0.0
            return x, y, z
    except Exception:
        pass
    return None

print("=== Starting Controlled Low-Altitude Flight ===", flush=True)

# Step 1: Unpause sim if paused
subprocess.run([
    "gz", "service", "-s", "/world/quadcopter/control",
    "--reqtype", "gz.msgs.WorldControl",
    "--reptype", "gz.msgs.Boolean",
    "--timeout", "2000",
    "--req", "pause: false"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

time.sleep(0.5)

target_alt = 1.8  # Target altitude in meters (stays clearly visible)
base_hover = 660.0

start_time = time.time()
flight_duration = 20  # Run for 20 seconds

print(f"Target Altitude: {target_alt}m (visible height)", flush=True)

# Simple closed loop on altitude + forward tilt after reaching 1.2m
while time.time() - start_time < flight_duration:
    state = get_state()
    if state is None:
        time.sleep(0.05)
        continue
    
    x, y, z = state
    alt_error = target_alt - z
    
    # Proportional controller on vertical thrust
    # Clamp throttle delta between -40 and +30 to prevent fast shooting up
    throttle_delta = max(min(alt_error * 35.0, 30.0), -40.0)
    w_base = base_hover + throttle_delta

    # Once lifted above 1.0m, pitch forward slightly:
    # rotor 0 (front-right) & rotor 2 (front-left) slightly lower
    # rotor 1 (rear-left) & rotor 3 (rear-right) slightly higher
    pitch_diff = 0.0
    if z > 0.8:
        pitch_diff = 3.5  # mild pitch forward
    
    w0 = w_base - pitch_diff
    w1 = w_base + pitch_diff
    w2 = w_base - pitch_diff
    w3 = w_base + pitch_diff

    send_motors(round(w0, 1), round(w1, 1), round(w2, 1), round(w3, 1))
    
    print(f"Time: {time.time()-start_time:4.1f}s | Pos: (x={x:5.2f}, y={y:5.2f}, z={z:5.2f}m) | Speed: {w_base:.1f} RPM", flush=True)
    time.sleep(0.05)

print("\nFlight routine completed! Stabilizing hover...", flush=True)
# Hold hover briefly
for _ in range(20):
    send_motors(660, 660, 660, 660)
    time.sleep(0.05)

print("Done! Drone flight finished.", flush=True)
