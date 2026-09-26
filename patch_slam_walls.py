import re

with open('map/nidar_slam_node.py', 'r') as f:
    content = f.read()

old_loop = """
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
"""

new_loop = """
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= msg.range_min:
                    continue
                ray_ang = (yaw + angle_min + i * angle_step + math.pi) % (2 * math.pi) - math.pi
                hit_wall = (r < max_r - 0.10)
                dist_ray = min(r, max_r)
                
                # Draw free space only up to 0.15m before the wall to prevent erasing walls!
                free_dist = max(0.0, dist_ray - 0.15) if hit_wall else dist_ray
                fwx = cx + free_dist * math.cos(ray_ang)
                fwy = cy + free_dist * math.sin(ray_ang)
                c_free = world_to_cell(fwx, fwy)
                
                if c_free:
                    cv2.line(free_mask, (x0, y0), (c_free[0], c_free[1]), 8, 1)
                
                if hit_wall:
                    wx = cx + dist_ray * math.cos(ray_ang)
                    wy = cy + dist_ray * math.sin(ray_ang)
                    c_end = world_to_cell(wx, wy)
                    if c_end:
                        x1, y1 = c_end
                        in_gate = (-6.75 <= wx <= -4.85 and -7.65 <= wy <= -7.35)
                        near_survivor = any(math.hypot(wx - s['x'], wy - s['y']) < 0.55 for s in self.confirmed_survivors)
                        if not in_gate and not near_survivor:
                            cv2.circle(wall_mask, (x1, y1), 0, 30, -1)
"""

if "free_mask[y1, x1] = 0" in content:
    content = content.replace(old_loop.strip(), new_loop.strip())
    with open('map/nidar_slam_node.py', 'w') as f:
        f.write(content)
        print("Patched walls logic successfully!")
else:
    print("Could not find the target code to patch!")

