import yaml
with open('config/nidar_map.rviz', 'r') as f:
    data = yaml.safe_load(f)

# Remove the PointCloud2 or other custom map displays
displays = data.get('Visualization Manager', {}).get('Displays', [])
displays = [d for d in displays if d.get('Name') not in ['Colorized Map', 'Map Free', 'Map Walls', 'Map Unknown']]

# Add 3 GridCells
displays.append({
    'Class': 'rviz_default_plugins/GridCells',
    'Enabled': True,
    'Name': 'Map Free',
    'Color': '180; 180; 180',
    'Topic': {'Depth': 5, 'Durability Policy': 'Transient Local', 'History Policy': 'Keep Last', 'Reliability Policy': 'Reliable', 'Value': '/map_cells_free'}
})
displays.append({
    'Class': 'rviz_default_plugins/GridCells',
    'Enabled': True,
    'Name': 'Map Walls',
    'Color': '10; 10; 10',
    'Topic': {'Depth': 5, 'Durability Policy': 'Transient Local', 'History Policy': 'Keep Last', 'Reliability Policy': 'Reliable', 'Value': '/map_cells_walls'}
})
displays.append({
    'Class': 'rviz_default_plugins/GridCells',
    'Enabled': True,
    'Name': 'Map Unknown',
    'Color': '0; 150; 255',
    'Topic': {'Depth': 5, 'Durability Policy': 'Transient Local', 'History Policy': 'Keep Last', 'Reliability Policy': 'Reliable', 'Value': '/map_cells_unknown'}
})

data['Visualization Manager']['Displays'] = displays

with open('config/nidar_map.rviz', 'w') as f:
    yaml.dump(data, f)
