#!/usr/bin/env python3
"""Test the parser fix on a single YAML file."""
import sys
import logging
from pathlib import Path

# Configure logging to see parser diagnostics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from boltz.data.parse.yaml import parse_yaml

def test_yaml(yaml_path: str):
    """Parse a YAML file and report results."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return False
    
    print(f"\nTesting: {path.name}")
    print("=" * 60)
    
    try:
        # Note: parse_yaml requires mols and mol_dir, but we can pass empty dicts
        # This will test the structure parsing and alignment without full molecule deps
        result = parse_yaml(path, ccd={}, mol_dir=Path('/tmp/test_mol_dir'), boltz2=True)
        print(f"✓ SUCCESS: Parsed {path.name}")
        if hasattr(result, 'chain_id_to_chain'):
            chains = result.chain_id_to_chain
            print(f"  Chains found: {list(chains.keys())}")
            for chain_id, chain_obj in chains.items():
                if hasattr(chain_obj, 'residues'):
                    print(f"    {chain_id}: {len(chain_obj.residues)} residues")
        return True
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: test_yaml_fix.py <yaml_file>")
        print("\nExample:")
        print("  python test_yaml_fix.py AA2AR_yaml_configs_test/CHEMBL190.yaml")
        sys.exit(1)
    
    success = test_yaml(sys.argv[1])
    sys.exit(0 if success else 1)
