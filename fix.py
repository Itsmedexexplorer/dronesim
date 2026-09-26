import os
d = '/home/dexter/Downloads/Drone-simulation-2-main'
for r, _, fs in os.walk(d):
    for f in fs:
        if f.startswith('.') or f.endswith('.png') or f.endswith('.pgm') or f.endswith('.zip') or f == 'fix.py': continue
        path = os.path.join(r, f)
        try:
            with open(path, 'r') as file:
                content = file.read()
            if '/home/priyansh/drone-sim' in content:
                content = content.replace('/home/priyansh/drone-sim', d)
                with open(path, 'w') as file:
                    file.write(content)
                print(f"Replaced in {path}")
        except Exception as e:
            pass
