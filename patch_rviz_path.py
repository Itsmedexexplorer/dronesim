import yaml
with open('config/nidar_map.rviz', 'r') as f:
    data = yaml.safe_load(f)

displays = data.get('Visualization Manager', {}).get('Displays', [])
# Remove existing Path if any, though likely not there
displays = [d for d in displays if d.get('Name') not in ['Nav2 Plan', 'Global Path']]

# Add the Nav2 Plan display
displays.append({
    'Class': 'rviz_default_plugins/Path',
    'Enabled': True,
    'Name': 'Nav2 Plan',
    'Color': '0; 255; 0',  # Solid Green
    'Line Style': 'Lines',
    'Line Width': 0.1,
    'Topic': {'Depth': 5, 'Durability Policy': 'Volatile', 'History Policy': 'Keep Last', 'Reliability Policy': 'Reliable', 'Value': '/plan'}
})

data['Visualization Manager']['Displays'] = displays

with open('config/nidar_map.rviz', 'w') as f:
    yaml.dump(data, f)
