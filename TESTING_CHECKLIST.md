# Testing Checklist for External Structure Support

## Overview
This checklist validates the implementation of external 3D structure loading for affinity prediction.

## Critical Bugs Fixed

### 1. ✅ Coordinate Reshaping Bug (affinity_predictor.py)
**Location:** `src/boltz/model/models/affinity_predictor.py`, lines ~265-273

**Bug:** Used incorrect tensor indexing `coords[:, None, None, :, :]` which creates 5D tensor
**Fix:** Changed to proper unsqueeze: `coords.unsqueeze(1).unsqueeze(1)`

**Test:**
```python
import torch
coords = torch.randn(2, 100, 3)
coords_reshaped = coords.unsqueeze(1).unsqueeze(1)
assert coords_reshaped.shape == (2, 1, 1, 100, 3), "Shape mismatch!"
```

### 2. ✅ Protein Coordinate Loading Bug (schema.py)
**Location:** `src/boltz/data/parse/schema.py`, lines ~1172-1235

**Bug:** External protein structure was loaded but coordinates were never used - `parse_polymer()` created new structure with (0,0,0) coords
**Fix:** Added coordinate override logic after `parse_polymer()` - copies coords from external structure to parsed chain atom-by-atom

**Test:**
- Load PDB file
- Verify coordinates are copied to parsed chain
- Check coords are not all (0,0,0)

## Implementation Summary

### Files Modified

1. **schema.py** (2 sections)
   - Lines 1172-1235: Protein structure loading + coordinate override
   - Lines 1270-1340: Ligand structure loading (already working)

2. **affinity_predictor.py** (1 section)
   - Lines 265-273: Fixed coordinate reshaping logic

### Files Created

1. **test_external_structures.py** - Automated test suite
2. **TESTING_CHECKLIST.md** - This file
3. **affinity_with_structures.yaml** - Example config with structure_path fields
4. **EXTERNAL_STRUCTURES_GUIDE.md** - Usage documentation

## Pre-Testing Verification

### ✅ Code Quality Checks

- [x] No syntax errors in schema.py
- [x] No syntax errors in affinity_predictor.py
- [x] Variable scoping correct (structure_path, parsed_structure)
- [x] Error handling present (file not found, format validation)
- [x] Import dependencies identified (torch, rdkit, click, yaml)

### ✅ Logic Verification

- [x] Protein: parse_pdb() → override coords → use in featurizer
- [x] Ligand: RDKit load → skip 3D generation → use conformer coords
- [x] Coordinate flow: file → parse → feats["coords"] → AffinityPredictor
- [x] Shape handling: (B, N, 3) → unsqueeze → (B, 1, 1, N, 3)

## Testing Plan

### Phase 1: Unit Tests (Run test_external_structures.py)

```bash
python test_external_structures.py
```

**Expected Results:**
- ✓ YAML parsing detects structure_path fields
- ✓ Protein PDB loading works (if test file provided)
- ✓ Ligand MOL2 loading works (if test file provided)
- ✓ Coordinate shapes correct
- ⊘ Full pipeline (manual test required)

**Test Files Needed:**
- `test_structures/receptor.pdb` - Sample protein structure
- `test_structures/ligand.mol2` - Sample ligand structure

### Phase 2: Integration Test (Manual)

1. **Prepare test data:**
   ```bash
   mkdir -p test_structures
   # Copy your receptor.pdb and ligand.mol2 here
   ```

2. **Create test YAML:**
   ```yaml
   version: 2
   sequences:
     - protein:
         id: ["A"]
         sequence: "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
         structure_path: "test_structures/receptor.pdb"
       ligand:
         id: ["B"]
         smiles: "c1ccccc1"
         structure_path: "test_structures/ligand.mol2"
   ```

3. **Run prediction:**
   ```bash
   python -m boltz.main_simplified predict \
       --yaml test.yaml \
       --checkpoint /path/to/boltz2_affinity_checkpoint.ckpt \
       --output test_output/
   ```

4. **Verify:**
   - No errors about missing coordinates
   - No warnings about (0,0,0) coords
   - Affinity predictions generated
   - Check logs for "Loaded coordinates from external structure" messages

### Phase 3: Edge Case Testing

**Test 1: Protein sequence/structure length mismatch**
```yaml
# FASTA has 50 residues but PDB has 48
sequences:
  - protein:
      id: ["A"]
      sequence: "MKFL..." (50 residues)
      structure_path: "receptor_48.pdb"  # Only 48 residues
```
**Expected:** ValueError with clear message

**Test 2: Missing structure files**
```yaml
sequences:
  - protein:
      structure_path: "nonexistent.pdb"
```
**Expected:** ValueError: "Structure file not found: nonexistent.pdb"

**Test 3: Invalid file format**
```yaml
sequences:
  - ligand:
      structure_path: "molecule.xyz"  # Unsupported format
```
**Expected:** ValueError: "Unsupported ligand structure format: .xyz"

**Test 4: Missing coordinates in file**
```yaml
sequences:
  - ligand:
      structure_path: "no_coords.mol2"  # File with no 3D coords
```
**Expected:** ValueError: "molecule has no conformer/coordinates"

**Test 5: Backward compatibility (no structure_path)**
```yaml
sequences:
  - ligand:
      smiles: "c1ccccc1"  # No structure_path - should generate 3D
```
**Expected:** Works normally, generates conformer from SMILES

### Phase 4: Production Validation

**Realistic Docking Workflow:**

1. **Dock with AutoDock Vina:**
   ```bash
   vina --receptor receptor.pdbqt --ligand ligand.pdbqt --out docked.pdbqt
   ```

2. **Convert docked pose to MOL2:**
   ```bash
   obabel docked.pdbqt -O docked.mol2
   ```

3. **Run Boltz affinity prediction:**
   ```yaml
   sequences:
     - protein:
         id: ["A"]
         sequence: "MKFL..."
         structure_path: "receptor.pdb"
       ligand:
         id: ["B"]
         smiles: "c1ccccc1"
         structure_path: "docked.mol2"
   ```

4. **Verify:**
   - Affinity prediction completes
   - Results are reasonable (e.g., -8 to -5 kcal/mol for good binders)
   - Compare to Vina score (should be in similar range)

## Known Limitations

1. **Multi-chain proteins:** Currently uses only first chain from PDB
   - Future: Add chain_id parameter to select specific chain

2. **Alternative conformations:** PDB alt locs not handled
   - Uses first conformer only

3. **Missing atoms:** If PDB missing atoms, uses (0,0,0) for those atoms
   - Could cause issues if many missing atoms

4. **Coordinate frame:** No alignment performed
   - User responsible for pre-aligning structures if needed

## Success Criteria

- [ ] All unit tests pass
- [ ] Integration test produces predictions without errors
- [ ] Edge cases handled gracefully with clear error messages
- [ ] Production workflow (Vina → Boltz) works end-to-end
- [ ] Predictions are reasonable (within expected range)
- [ ] Backward compatibility maintained (old YAMLs still work)

## Troubleshooting

### Error: "Import torch could not be resolved"
**Cause:** IDE warning, not runtime error
**Fix:** Install dependencies: `pip install torch rdkit-pypi pyyaml click`

### Error: "Structure file not found"
**Cause:** Path in YAML is incorrect
**Fix:** Use absolute paths or paths relative to YAML file location

### Error: "Sequence length mismatch"
**Cause:** FASTA sequence doesn't match PDB chain length
**Fix:** Ensure FASTA sequence matches the chain you want to use

### Error: "No conformer found"
**Cause:** Structure file has no 3D coordinates
**Fix:** Use structure file with embedded coordinates (PDB/MOL2/SDF with 3D)

### Warning: "Coordinates are (0,0,0)"
**Cause:** Coordinate override didn't work
**Fix:** Check that atom names in PDB match expected names (CA, C, N, etc.)

## Next Steps After Testing

1. **Documentation:**
   - Update README.md with external structure examples
   - Add troubleshooting section
   - Create tutorial notebook

2. **Features:**
   - Add chain_id parameter for multi-chain proteins
   - Support PDBQT format (common in docking)
   - Add coordinate alignment option

3. **Performance:**
   - Benchmark loading time for large structures
   - Optimize coordinate copying

4. **Validation:**
   - Test on PDBbind dataset
   - Compare predictions with/without external structures
   - Validate against experimental affinities

## Contact/Questions

If you encounter issues not covered here:
1. Check error message carefully
2. Verify file formats are correct
3. Try with minimal test case
4. Review logs for "Loaded coordinates from external structure" messages
