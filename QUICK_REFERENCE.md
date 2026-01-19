# Quick Reference: External Structure Loading

## YAML Syntax

```yaml
version: 2
sequences:
  - protein:
      id: ["A"]
      sequence: "MKFLKF..."
      structure_path: "receptor.pdb"  # ← NEW FIELD
      
  - ligand:
      id: ["B"]
      smiles: "c1ccccc1"
      structure_path: "ligand.mol2"   # ← NEW FIELD
```

## Supported Formats

| Type | Formats | Notes |
|------|---------|-------|
| Protein | .pdb, .cif, .mmcif | Must have ATOM records |
| Ligand | .mol2, .sdf, .pdb | Must have 3D coordinates |

## Critical Requirements

1. **Sequence must match structure**
   - FASTA residue count = PDB residue count
   
2. **3D coordinates required**
   - File must contain embedded coordinates
   - Not just connectivity/topology
   
3. **Paths can be absolute or relative**
   - Relative to YAML file location

## Quick Test

```bash
# 1. Create test config
cat > test.yaml << EOF
version: 2
sequences:
  - protein:
      id: ["A"]
      sequence: "MKFLKF"  # Match your PDB
      structure_path: "my_receptor.pdb"
  - ligand:
      id: ["B"] 
      smiles: "c1ccccc1"
      structure_path: "my_ligand.mol2"
EOF

# 2. Run prediction
python -m boltz.main_simplified predict \
    --yaml test.yaml \
    --checkpoint checkpoint.ckpt \
    --output output/

# 3. Check for success message
grep "Loaded coordinates from external structure" output/log.txt
```

## Expected Log Output

```
Loading protein structure from my_receptor.pdb
Loaded 1 chain(s) from my_receptor.pdb
Loaded coordinates from external structure for 248 residues
Loaded ligand structure from my_ligand.mol2 (32 atoms)
```

## Common Errors

| Error Message | Fix |
|--------------|-----|
| Structure file not found | Check path is correct |
| Unsupported format | Use .pdb/.mol2/.sdf |
| No conformer | Add 3D coords to file |
| Sequence length mismatch | Match FASTA to structure |

## Backward Compatibility

Old YAMLs without `structure_path` still work:
```yaml
# This still works (generates 3D coords)
sequences:
  - ligand:
      smiles: "c1ccccc1"  # No structure_path
```

## Files to Check

- [READY_FOR_TESTING.md](READY_FOR_TESTING.md) - Full implementation details
- [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) - Comprehensive testing guide  
- [test_external_structures.py](test_external_structures.py) - Automated tests
- [examples/affinity_with_structures.yaml](examples/affinity_with_structures.yaml) - Full example
