import json
with open('preprocess_full_study_step_by_step.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'ajk_tehsils' in source.lower() or 'gb_' in source.lower() or 'kpk' in source.lower():
            print(source)
            print('---')
