#!/usr/bin/env python3
"""Diagnostic: Compare YAML sequence vs PDB structure sequence."""
import sys
from pathlib import Path
import gemmi

# Read YAML sequence
yaml_file = Path("AA2AR_yaml_configs_test/CHEMBL190.yaml")
if yaml_file.exists():
    import yaml
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    yaml_seq = None
    for item in data.get("sequences", []):
        if "protein" in item:
            yaml_seq = item["protein"]["sequence"]
            break
    print(f"YAML sequence (length {len(yaml_seq) if yaml_seq else 0}):")
    if yaml_seq:
        print(f"  {yaml_seq[:80]}...")
        print(f"  First 10 aa: {yaml_seq[:10]}")
else:
    print(f"YAML file not found: {yaml_file}")
    yaml_seq = None

# Read PDB structure
pdb_file = Path("receptors/AA2AR_rec.pdb")
if pdb_file.exists():
    structure = gemmi.read_structure(str(pdb_file))
    structure.setup_entities()
    
    print(f"\nPDB structure: {pdb_file}")
    print(f"  Models: {len(structure)}")
    print(f"  Entities: {len(structure.entities)}")
    
    for entity in structure.entities:
        if entity.entity_type.name == "Polymer":
            print(f"\nEntity: {entity.name}")
            print(f"  Type: {entity.polymer_type.name}")
            print(f"  Subchains: {entity.subchains}")
            print(f"  Full sequence length: {len(entity.full_sequence)}")
            print(f"  Full sequence: {entity.full_sequence}")
            
            # Get first chain residues
            for chain in structure[0]:
                for res in chain:
                    if res.subchain == entity.subchains[0]:
                        residues = [r.name for r in chain if r.subchain == entity.subchains[0]]
                        print(f"  Chain residues (first 20): {residues[:20]}")
                        print(f"  Total residues in structure: {len(residues)}")
                        break
                break
else:
    print(f"PDB file not found: {pdb_file}")

# Compare
if yaml_seq:
    from boltz.data import const
    token_map = const.prot_letter_to_token
    unk_token = const.unk_token["PROTEIN"]
    seq_tokens = [token_map.get(c, unk_token) for c in list(yaml_seq)]
    print(f"\nYAML sequence as tokens (first 20): {seq_tokens[:20]}")
