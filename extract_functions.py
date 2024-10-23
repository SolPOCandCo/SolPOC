#%%
with open('solpoc/functions_SolPOC.py', encoding='utf-8') as f:
    script = f.read()

import re

matches = re.findall(r'def (.+)\s*\(',script)

out = '\n'.join(
    sorted(
        map(lambda s: f'from .functions_SolPOC import {s}',
        filter(lambda x: not x.startswith('_'), matches)
        ),
    key=str.casefold)
)

print(out)
with open('funcs_for_init.tmp', 'w') as f:
    f.write(out)
