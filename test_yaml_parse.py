#!/usr/bin/env python
"""Test YAML parsing."""
import sys
sys.path.insert(0, '/Users/limcaoco/Projects/dock_boltz/src')

from pathlib import Path
from boltz.data.parse.yaml import parse_yaml
from boltz.main_simplified import load_mols
import traceback

try:
    # Load molecules
    print('Loading canonical molecules...')
    mols = load_mols()
    print(f'  Loaded {len(mols)} molecules')
    
    mol_dir = Path('/tmp/test_mol_dir')
    mol_dir.mkdir(exist_ok=True)
    
    # Parse YAML
    print('Parsing YAML...')
    result = parse_yaml(Path('AA2AR_yaml_configs_test/CHEMBL100382.yaml'), mols, mol_dir, boltz2=True)
    print('✓ YAML parsed successfully!')
    print(f'  Chains: {list(result.chain_id_to_chain.keys())}')
    if result.record.affinity:
        print(f'  Affinity ligand: {result.record.affinity}')
    else:
        print(f'  No affinity target')
except Exception as e:
    print(f'✗ Error: {type(e).__name__}: {e}')
    traceback.print_exc()
