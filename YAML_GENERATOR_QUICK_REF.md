# YAML Generator Quick Reference

Fast guide to using `scripts/generate_affinity_yaml.py` for batch scoring.

## Installation

```bash
# Install dependencies
pip install pyyaml rdkit-pypi

# For auto-extraction of protein sequence from PDB (optional but recommended):
pip install biopython

# Make script executable
chmod +x scripts/generate_affinity_yaml.py
```

## Basic Usage

### Option 1: Provide sequence explicitly

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output batch_affinity.yaml
```

### Option 2: Auto-extract sequence from PDB (requires BioPython)

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output batch_affinity.yaml
```

**When to use:** Easiest option if you have the PDB file and don't have FASTA

### Option 3: Auto-extract both sequence and SMILES

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --poses-dir ./poses/ \
    --auto-smiles \
    --output batch_affinity.yaml
```

**When to use:** Full automation - all you need is PDB + pose directory

### Individual YAMLs (one per pose)

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output-dir ./yaml_configs/
```

**When to use:** Large batches, parallel processing, job scheduling

## Common Scenarios

### Scenario 1: Vina docking (auto-extract sequence)

```bash
# 1. Run Vina and get docked.pdbqt
vina --receptor receptor.pdbqt --ligand ligand.pdbqt --out docked.pdbqt --num_modes 20

# 2. Convert poses
obabel docked.pdbqt -O poses/pose_%04d.mol2 -m

# 3. Generate YAML (sequence auto-extracted from receptor.pdb)
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --ligand "c1ccc(cc1)O" \
    --poses-dir poses/ \
    --output affinity_batch.yaml

# 4. Run predictions
python -m boltz.main_simplified predict \
    --yaml affinity_batch.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/
```

### Scenario 2: Auto-detect SMILES and sequence

```bash
# If you only have receptor.pdb and pose directory
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --poses-dir ./poses/ \
    --auto-smiles \
    --output batch_affinity.yaml
```

### Scenario 3: Different file formats

```bash
# For SDF files
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --pattern "*.sdf" \
    --output batch_affinity.yaml

# For PDB files
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --pattern "*.pdb" \
    --output batch_affinity.yaml
```

### Scenario 4: Skip validation (faster, risky)

```bash
# For large batches where you know files are valid
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --no-validate \
    --output batch_affinity.yaml
```

## Argument Reference

```
REQUIRED:
  --protein PROTEIN_PDB          Protein structure file (PDB/mmCIF)
  --poses-dir DIRECTORY          Directory with pose files

PROTEIN SEQUENCE (choose one):
  --protein-seq SEQUENCE         Provide sequence directly
  (auto-extract from PDB)        If not provided, auto-extract from PDB (needs BioPython)

LIGAND (choose one):
  --ligand SMILES                Provide SMILES directly
  --auto-smiles                  Extract from first pose

OUTPUT (choose one or both):
  --output FILE.yaml             Single YAML with all poses
  --output-dir DIRECTORY         Separate YAML per pose

OPTIONAL:
  --pattern "*.mol2"             File pattern (default: *.mol2)
  --no-validate                  Skip file validation
  --verbose                      Verbose output
```

## Output Verification

### Check single YAML
```bash
# Count sequences
grep "^  - protein:" batch_affinity.yaml | wc -l

# View structure
head -20 batch_affinity.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('batch_affinity.yaml'))"
```

### Check individual YAMLs
```bash
# Count files
ls yaml_configs/*.yaml | wc -l

# Check first file
head -20 yaml_configs/pose_0000_*.yaml
```

## Workflow Templates

### Template 1: Quick test (small batch)

```bash
#!/bin/bash

PROTEIN="target.pdb"
PROTEIN_SEQ="MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
LIGAND_SMILES="c1ccccc1"
POSES_DIR="./poses"
### Template 1: Quick test (small batch) - Auto-extract sequence

```bash
#!/bin/bash

PROTEIN="target.pdb"
LIGAND_SMILES="c1ccccc1"
POSES_DIR="./poses"
OUTPUT="affinity_batch.yaml"
CHECKPOINT="/path/to/checkpoint.ckpt"

# Generate YAML (sequence auto-extracted from PDB)
python scripts/generate_affinity_yaml.py \
    --protein "$PROTEIN" \
    --ligand "$LIGAND_SMILES" \
    --poses-dir "$POSES_DIR" \
    --output "$OUTPUT"

# Run predictions
python -m boltz.main_simplified predict \
    --yaml "$OUTPUT" \
    --checkpoint "$CHECKPOINT" \
    --output results/

# Show results
head -20 results/predictions.yaml
```

### Template 2: Large batch (parallel) - Auto-extract everything

```bash
#!/bin/bash

PROTEIN="target.pdb"
POSES_DIR="./poses"
YAML_DIR="./yaml_configs"
CHECKPOINT="/path/to/checkpoint.ckpt"
NUM_JOBS=4

# Generate individual YAMLs (sequence + SMILES auto-extracted)
python scripts/generate_affinity_yaml.py \
    --protein "$PROTEIN" \
    --poses-dir "$POSES_DIR" \
    --auto-smiles \
    --output-dir "$YAML_DIR"

# Run in parallel
echo "Processing $(ls $YAML_DIR/*.yaml | wc -l) poses with $NUM_JOBS parallel jobs..."
ls $YAML_DIR/*.yaml | \
    xargs -P $NUM_JOBS -I {} \
    python -m boltz.main_simplified predict \
        --yaml {} \
        --checkpoint "$CHECKPOINT" \
        --output results/{/.}

echo "Done! Results in results/"
```

### Template 3: Vina → Boltz pipeline

```bash
#!/bin/bash

TARGET="target"
LIGAND_SMILES="c1ccccc1"
CHECKPOINT="/path/to/checkpoint.ckpt"
VINA_MODES=20
BOLTZ_GPU="cuda"

# 1. Run Vina
vina \
    --receptor docking/${TARGET}.pdbqt \
    --ligand docking/ligand.pdbqt \
    --out docking/docked.pdbqt \
    --num_modes $VINA_MODES

# 2. Convert poses
mkdir -p poses
obabel docking/docked.pdbqt -O poses/pose_%04d.mol2 -m

# 3. Get sequence
SEQ=$(tail -1 ${TARGET}.fasta)

# 4. Generate YAML
python scripts/generate_affinity_yaml.py \
    --protein ${TARGET}.pdb \
    --protein-seq "$SEQ" \
    --ligand "$LIGAND_SMILES" \
    --poses-dir poses/ \
    --output affinity_batch.yaml

# 5. Score with Boltz
mkdir -p affinity_results
python -m boltz.main_simplified predict \
    --yaml affinity_batch.yaml \
    --checkpoint "$CHECKPOINT" \
    --output affinity_results/ \
    --device "$BOLTZ_GPU"

# 6. Compare scores
python << 'EOF'
import yaml, pandas as pd

# Load Boltz results
with open("affinity_results/predictions.yaml") as f:
    boltz_data = yaml.safe_load(f)

df = pd.DataFrame([
    {"pose": p.get("name", "unknown"), "boltz_affinity": p.get("metrics", {}).get("affinity")}
    for p in boltz_data
])

df = df.sort_values("boltz_affinity")
df.to_csv("affinity_results/ranked_poses.csv", index=False)
print(df)
EOF
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No pose files matching pattern found" | Check `ls poses/` - are files there? Update `--pattern` |
| "Could not load molecule" | Files corrupted or wrong format - re-convert with obabel |
| "No 3D coordinates" | Pose file missing 3D coords - add them or regenerate |
| "Sequence length mismatch" | FASTA doesn't match PDB - verify sequence is correct |
| Script is slow | Use `--no-validate` or split into individual YAMLs for parallel |

## Examples

See [examples/batch_scoring_example/](../examples/batch_scoring_example/) for complete working example.

## Next: Run Predictions

Once you have the YAML:

```bash
python -m boltz.main_simplified predict \
    --yaml batch_affinity.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/ \
    --device cuda  # Add if GPU available
```

Results will include affinity scores for each pose. See [BATCH_AFFINITY_SCORING.md](../BATCH_AFFINITY_SCORING.md) for post-processing examples.
