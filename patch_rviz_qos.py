import yaml
with open('config/nidar_map.rviz', 'r') as f:
    data = yaml.safe_load(f)

for display in data.get('Visualization Manager', {}).get('Displays', []):
    if display.get('Name') == 'Colorized Map':
        display['Topic'] = {
            'Depth': 5,
            'Durability Policy': 'Transient Local',
            'History Policy': 'Keep Last',
            'Reliability Policy': 'Reliable',
            'Value': '/map_color_cloud'
        }

with open('config/nidar_map.rviz', 'w') as f:
    yaml.dump(data, f)
