#!/usr/bin/env python3
"""Check all YAML files in a directory using boltz parser and report failures."""
from pathlib import Path
import sys
from boltz.main_simplified import load_mols
from boltz.data.parse.yaml import parse_yaml


def main(dir_path: str):
    mols = load_mols()
    mol_dir = Path('/tmp/test_mol_dir')
    mol_dir.mkdir(exist_ok=True)
    dirp = Path(dir_path)
    files = sorted(dirp.glob('*.yaml'))
    if not files:
        print('No YAML files found in', dirp)
        return
    for f in files:
        try:
            print('Parsing', f.name)
            parse_yaml(f, mols, mol_dir, boltz2=True)
        except Exception as e:
            print('FAILED', f.name, type(e).__name__, e)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: check_yamls.py <yaml-dir>')
        sys.exit(1)
    main(sys.argv[1])
