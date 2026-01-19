# External Structure Support - Implementation Summary

## Overview

Added support for using external 3D structure files (PDB, MOL2, SDF, etc.) in the YAML input format, allowing users to provide their own protein and ligand coordinates from docking tools, molecular dynamics, or other structure prediction methods.

## Changes Made

### 1. Modified YAML Schema Parser

**File:** [`src/boltz/data/parse/schema.py`](src/boltz/data/parse/schema.py)

**Changes:**

#### For Ligands (lines ~1234-1280):
```python
# Added structure_path field support
structure_path = items[0][entity_type].get("structure_path", None)

if structure_path is not None:
    # Load from file instead of generating from SMILES
    if structure_path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(structure_path), removeHs=False)
    elif structure_path.suffix.lower() in [".sdf", ".sd"]:
        supplier = Chem.SDMolSupplier(str(structure_path), removeHs=False)
        mol = supplier[0]
    elif structure_path.suffix.lower() == ".pdb":
        mol = Chem.MolFromPDBFile(str(structure_path), removeHs=False)
    
    # Skip 3D conformer generation
else:
    # Original behavior: generate from SMILES
    mol = AllChem.MolFromSmiles(seq)
    success = compute_3d_conformer(mol)
```

#### For Proteins (lines ~1168-1195):
```python
# Added structure_path field support
structure_path = items[0][entity_type].get("structure_path", None)

if structure_path is not None and entity_type == "protein":
    # Load structure from PDB/mmCIF
    if structure_path.suffix.lower() == ".pdb":
        parsed_structure = parse_pdb(structure_path)
    elif structure_path.suffix.lower() in [".cif", ".mmcif"]:
        parsed_structure = parse_mmcif(structure_path)
    
    # Coordinates preserved from loaded file
```

### 2. Created Example Config

**File:** [`examples/affinity_with_structures.yaml`](examples/affinity_with_structures.yaml)

Shows the new YAML format with `structure_path` fields.

### 3. Updated Documentation

**Files Updated:**
- [`USAGE_EXAMPLES.md`](USAGE_EXAMPLES.md) - Added examples for using external structures
- [`README.md`](README.md) - Highlighted new feature in quick start

## New YAML Format

### Basic Format

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ...
      structure_path: /path/to/protein.pdb  # Optional
      
  - ligand:
      id: B
      smiles: 'CCO'
      structure_path: /path/to/ligand.mol2  # Optional

properties:
  - affinity:
      binder: B
```

### Supported File Formats

| Molecule Type | Formats | RDKit Loader |
|--------------|---------|--------------|
| Protein | `.pdb` | `MolFromPDBFile()` |
| Protein | `.cif`, `.mmcif` | `parse_mmcif()` |
| Ligand | `.pdb` | `MolFromPDBFile()` |
| Ligand | `.mol2` | `MolFromMol2File()` |
| Ligand | `.sdf`, `.sd` | `SDMolSupplier()` |

## Usage Examples

### From AutoDock Vina

```bash
# Convert PDBQT to supported formats
obabel receptor.pdbqt -O receptor.pdb
obabel ligand_out.pdbqt -O ligand.mol2

# Create config
cat > vina_complex.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_SEQUENCE
      structure_path: receptor.pdb
  - ligand:
      id: B
      smiles: 'YOUR_SMILES'
      structure_path: ligand.mol2
properties:
  - affinity:
      binder: B
EOF

# Predict
python -m boltz.main_simplified predict \
    --input vina_complex.yaml \
    --output results.csv \
    --checkpoint model.ckpt
```

### From Glide/Schrödinger

```bash
# Convert MAE to MOL2
$SCHRODINGER/utilities/structconvert pose.mae pose.mol2

# Use in YAML
# structure_path: pose.mol2
```

### From AlphaFold3

```bash
# Use AF3 prediction directly
cat > af3_config.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_SEQUENCE
      structure_path: af3_prediction.pdb
  - ligand:
      id: B
      smiles: 'YOUR_SMILES'
      structure_path: ligand_from_af3.sdf
properties:
  - affinity:
      binder: B
EOF
```

## Backward Compatibility

The changes are **fully backward compatible**:

- ✅ Old YAML files (without `structure_path`) work unchanged
- ✅ Coordinates still generated from SMILES if no `structure_path` provided
- ✅ All existing functionality preserved

## Error Handling

The implementation includes validation:

1. **File existence check**: Raises error if file not found
2. **Format validation**: Only allows supported file extensions
3. **Coordinate verification**: Ensures loaded molecules have conformers
4. **Size limits**: Checks ligand atom count (128 max, 56 recommended)
5. **User feedback**: Prints confirmation messages when loading structures

## Testing

To test the new feature:

```bash
# 1. Create test structure files
# protein.pdb - standard PDB file
# ligand.mol2 - MOL2 with coordinates

# 2. Create config with structure paths
cat > test.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNV...
      structure_path: protein.pdb
  - ligand:
      id: B
      smiles: 'CCO'
      structure_path: ligand.mol2
properties:
  - affinity:
      binder: B
EOF

# 3. Validate
python -m boltz.main_simplified validate --input test.yaml

# 4. Run prediction
python -m boltz.main_simplified predict \
    --input test.yaml \
    --output results.csv \
    --checkpoint model.ckpt
```

## Future Enhancements

Potential improvements for later:

1. **Multi-conformer support**: Accept SDF files with multiple conformers
2. **Complex loading**: Support pre-assembled protein-ligand complexes
3. **Format auto-detection**: Infer format from content instead of extension
4. **Coordinate validation**: Check for clashes, missing atoms, etc.
5. **Alternative coordinate sources**: Support other structure databases (PDBbind, etc.)

## Summary

This implementation enables seamless integration of Boltz-2's affinity predictor with existing computational chemistry workflows. Users can now:

- Use docked poses from AutoDock, Glide, GOLD, etc.
- Leverage MD simulation snapshots
- Integrate AlphaFold3 or other structure prediction outputs
- Mix-and-match: external structures for some molecules, generated for others

The `structure_path` field is intuitive, optional, and backward-compatible—making it easy to adopt while preserving existing workflows.
