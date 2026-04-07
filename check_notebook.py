import json

with open('cse499b-skincon-herb-final-training.ipynb', 'r') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    if cell_type == 'markdown':
        source = ''.join(cell['source'])[:50]
        print(f'Cell {i}: {cell_type} - {source}')
    elif cell_type == 'code':
        source = ''.join(cell['source'])[:80]
        print(f'Cell {i}: {cell_type} - {source}')
