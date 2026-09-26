import yaml
with open('config/nidar_map.rviz', 'r') as f:
    data = yaml.safe_load(f)

displays = data.get('Visualization Manager', {}).get('Displays', [])
# Check if Camera already exists
if not any(d.get('Name') == 'Drone Camera' for d in displays):
    displays.append({
        'Class': 'rviz_default_plugins/Image',
        'Enabled': True,
        'Name': 'Drone Camera',
        'Topic': {'Value': '/nidar/camera_view', 'Reliability Policy': 'Reliable'}
    })
    
data['Visualization Manager']['Displays'] = displays

with open('config/nidar_map.rviz', 'w') as f:
    yaml.dump(data, f)
