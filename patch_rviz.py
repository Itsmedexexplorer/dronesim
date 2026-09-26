import yaml
with open('config/nidar_map.rviz', 'r') as f:
    data = yaml.safe_load(f)
for display in data.get('Visualization Manager', {}).get('Displays', []):
    if display.get('Name') == 'Occupancy Grid':
        display['Enabled'] = False
# Add PointCloud2 display
data['Visualization Manager']['Displays'].append({
    'Class': 'rviz_default_plugins/PointCloud2',
    'Enabled': True,
    'Name': 'Colorized Map',
    'Topic': {'Value': '/map_color_cloud'},
    'Style': 'Squares',
    'Size (m)': 0.05,
    'Color Transformer': 'RGB8',
})
with open('config/nidar_map.rviz', 'w') as f:
    yaml.dump(data, f)
