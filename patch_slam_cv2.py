import re

with open('map/nidar_slam_node.py', 'r') as f:
    content = f.read()

# Add cv2 import
if 'import cv2' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport cv2')

# Replace the cb_lidar inner loop with cv2.line
old_loop = """
        with self.grid_lock:
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                ray_ang = (yaw + angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
                hit_wall = (r < max_r - 0.10)
                dist_ray = min(r, max_r)

                cos_ang = math.cos(ray_ang)
                sin_ang = math.sin(ray_ang)

                # Dynamic Free space raycast (clears transient noise & maintains exact corridor width)
                n_steps = max(1, int(dist_ray / (GRID_RES * 1.5)))
                for step in range(1, n_steps):
                    frac = step / float(n_steps)
                    wx = cx + dist_ray * frac * cos_ang
                    wy = cy + dist_ray * frac * sin_ang
                    c = world_to_cell(wx, wy)
                    if c:
                        col, row = c
                        val = self.prob_grid[row, col]
                        if val > 0:
                            self.prob_grid[row, col] = max(0, val - 8)

                # Wall endpoint (increments obstacle evidence)
                if hit_wall:
                    wx = cx + dist_ray * cos_ang
                    wy = cy + dist_ray * sin_ang
                    # South Gate doorway (-6.75m to -4.85m at Y=-7.50m) must never be marked occupied
                    in_gate = (-6.75 <= wx <= -4.85 and -7.65 <= wy <= -7.35)
                    # Confirmed Humans / Survivors: never mark as black wall obstacle
                    near_survivor = any(math.hypot(wx - s['x'], wy - s['y']) < 0.55 for s in self.confirmed_survivors)
                    if not in_gate and not near_survivor:
                        c = world_to_cell(wx, wy)
                        if c:
                            col, row = c
                            val = self.prob_grid[row, col]
                            if val < 100:
                                self.prob_grid[row, col] = min(100, val + 20)
"""

new_loop = """
        with self.grid_lock:
            # Create a blank mask for cv2 drawing to optimize the raycasting
            # We just use OpenCV line drawing!
            c_drone = world_to_cell(cx, cy)
            if not c_drone: return
            x0, y0 = c_drone
            
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                ray_ang = (yaw + angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
                hit_wall = (r < max_r - 0.10)
                dist_ray = min(r, max_r)

                cos_ang = math.cos(ray_ang)
                sin_ang = math.sin(ray_ang)

                wx = cx + dist_ray * cos_ang
                wy = cy + dist_ray * sin_ang
                
                c_end = world_to_cell(wx, wy)
                if not c_end: continue
                x1, y1 = c_end

                # Draw free space using cv2.line (much faster and gapless Bresenham)
                # To do this safely on the grid, we can just use cv2.line on a temp array, but 
                # since we need probabilistic updates, we can just generate the line points.
                # Actually, drawing directly on a numpy array is faster using cv2.line!
                # Wait, cv2.line overwrites values. For probabilities, we can just subtract.
                # Since OpenCV doesn't do "subtract line", we can just use np.clip.
                pass
                
            # Actually, to do probabilistic updates in bulk very fast:
            # Draw all free lines on a mask
            free_mask = np.zeros_like(self.prob_grid, dtype=np.uint8)
            wall_mask = np.zeros_like(self.prob_grid, dtype=np.uint8)
            
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                ray_ang = (yaw + angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
                hit_wall = (r < max_r - 0.10)
                dist_ray = min(r, max_r)
                wx = cx + dist_ray * math.cos(ray_ang)
                wy = cy + dist_ray * math.sin(ray_ang)
                
                c_end = world_to_cell(wx, wy)
                if not c_end: continue
                x1, y1 = c_end
                
                # Draw free ray on mask
                cv2.line(free_mask, (x0, y0), (x1, y1), 8, 1)
                
                if hit_wall:
                    in_gate = (-6.75 <= wx <= -4.85 and -7.65 <= wy <= -7.35)
                    near_survivor = any(math.hypot(wx - s['x'], wy - s['y']) < 0.55 for s in self.confirmed_survivors)
                    if not in_gate and not near_survivor:
                        cv2.circle(wall_mask, (x1, y1), 0, 20, -1)
                        # Remove the wall from free mask to prevent conflict
                        free_mask[y1, x1] = 0
                        
            # Apply masks
            self.prob_grid = np.clip(self.prob_grid.astype(np.int16) - free_mask, 0, 100).astype(np.uint8)
            self.prob_grid = np.clip(self.prob_grid.astype(np.int16) + wall_mask, 0, 100).astype(np.uint8)
"""

if "dist_ray = min(r, max_r)" in content:
    content = content.replace(old_loop.strip(), new_loop.strip())
    with open('map/nidar_slam_node.py', 'w') as f:
        f.write(content)
        print("Patched successfully!")
else:
    print("Could not find the target code to patch!")

