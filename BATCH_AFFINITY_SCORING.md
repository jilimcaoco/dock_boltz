# Batch Pose Affinity Scoring

Generate YAML configs for scoring multiple docked poses with the Boltz-2 affinity module.

## Quick Start

### Single YAML with All Poses

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV \
    --ligand "c1ccc(cc1)O" \
    --poses-dir ./docked_poses/ \
    --output batch_affinity.yaml

# Run predictions on all poses
python -m boltz.main_simplified predict \
    --yaml batch_affinity.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/
```

### Individual YAMLs per Pose

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV \
    --ligand "c1ccc(cc1)O" \
    --poses-dir ./docked_poses/ \
    --output-dir yaml_configs/

# Process in parallel
parallel python -m boltz.main_simplified predict --yaml {} --checkpoint checkpoint.ckpt --output results/ ::: yaml_configs/*.yaml
```

## Typical Workflow

### 1. Dock ligand with AutoDock Vina

```bash
vina \
    --receptor receptor.pdbqt \
    --ligand ligand.pdbqt \
    --center_x 10.0 \
    --center_y 20.0 \
    --center_z 30.0 \
    --size_x 20 \
    --size_y 20 \
    --size_z 20 \
    --out docked.pdbqt \
    --num_modes 20
```

### 2. Convert docked poses to MOL2

```bash
# Split poses if needed
obabel docked.pdbqt -O pose_*.mol2 -m

# Or convert single pose
obabel docked.pdbqt -O docked.mol2
```

### 3. Generate YAML configs

```bash
# Option A: Get SMILES from structure
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "$(cat receptor.fasta)" \
    --poses-dir ./ \
    --auto-smiles \
    --output affinity_batch.yaml

# Option B: Provide SMILES explicitly
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "$(cat receptor.fasta)" \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --pattern "*.mol2" \
    --output affinity_batch.yaml
```

### 4. Run affinity predictions

```bash
python -m boltz.main_simplified predict \
    --yaml affinity_batch.yaml \
    --checkpoint checkpoint.ckpt \
    --output affinity_results/
```

### 5. Extract and rank predictions

```bash
python -c "
import yaml
import pandas as pd

with open('affinity_results/predictions.yaml') as f:
    results = yaml.safe_load(f)

# Create dataframe with affinity scores
data = []
for r in results:
    data.append({
        'pose': r['name'],
        'affinity': r['metrics']['affinity'],
    })

df = pd.DataFrame(data).sort_values('affinity')
print(df)
print(df.to_csv(index=False))
"
```

## Command Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--protein` | Path to protein structure (PDB/mmCIF) |
| `--protein-seq` | Protein sequence (FASTA) |
| `--poses-dir` | Directory containing docked poses |

### Ligand SMILES (choose one)

| Argument | Description |
|----------|-------------|
| `--ligand SMILES` | Provide SMILES explicitly |
| `--auto-smiles` | Extract SMILES from first pose file |

### Output (choose one or both)

| Argument | Description |
|----------|-------------|
| `--output FILE` | Generate single YAML with all poses |
| `--output-dir DIR` | Generate individual YAML per pose |

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--pattern` | Glob pattern for pose files (default: `*.mol2`) |
| `--no-validate` | Skip validation of structure files |
| `--verbose` | Verbose output |

## Supported File Formats

### Protein
- `.pdb` - PDB format
- `.cif`, `.mmcif` - mmCIF format

### Ligand/Poses
- `.mol2` - MOL2 format
- `.sdf`, `.sd` - SDF format
- `.pdb` - PDB format

All formats must contain embedded 3D coordinates.

## Examples

### Example 1: Vina docking → scoring

```bash
# 1. Run Vina
vina --receptor target.pdbqt --ligand ligand.pdbqt --out docked.pdbqt --num_modes 10

# 2. Convert poses
obabel docked.pdbqt -O pose*.mol2 -m

# 3. Create target.fasta file with sequence
echo ">target" > target.fasta
# Paste sequence...

# 4. Generate YAML
python scripts/generate_affinity_yaml.py \
    --protein target.pdb \
    --protein-seq "$(cat target.fasta | tail -1)" \
    --ligand "c1ccc(cc1)O" \
    --poses-dir ./ \
    --pattern "pose*.mol2" \
    --output vina_scoring.yaml

# 5. Score all poses
python -m boltz.main_simplified predict \
    --yaml vina_scoring.yaml \
    --checkpoint boltz2_affinity.ckpt \
    --output vina_scores/
```

### Example 2: Multiple ligands

Create separate YAML for each ligand:

```bash
for ligand in ligand*.mol2; do
    python scripts/generate_affinity_yaml.py \
        --protein target.pdb \
        --protein-seq "MKFLKF..." \
        --ligand "c1ccccc1" \
        --poses-dir "./poses_$ligand/" \
        --output "affinity_$ligand.yaml"
done

# Run all predictions
for yaml in affinity_*.yaml; do
    python -m boltz.main_simplified predict \
        --yaml "$yaml" \
        --checkpoint checkpoint.ckpt \
        --output "results_${yaml%.yaml}/"
done
```

### Example 3: Parallel processing

```bash
# Generate individual YAMLs
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output-dir ./yamls/

# Process in parallel with GNU parallel
parallel -j 4 python -m boltz.main_simplified predict \
    --yaml {} \
    --checkpoint checkpoint.ckpt \
    --output results/{/.}/ \
    ::: yamls/*.yaml

# Or with xargs
find yamls/ -name "*.yaml" | xargs -P 4 -I {} \
    python -m boltz.main_simplified predict \
        --yaml {} \
        --checkpoint checkpoint.ckpt \
        --output results/
```

## Output Format

### Single YAML Output (batch_affinity.yaml)

```yaml
version: 2
sequences:
  - protein:
      id: ["A"]
      sequence: MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV
      structure_path: receptor.pdb
    ligand:
      id: ["B"]
      smiles: c1ccc(cc1)O
      structure_path: ./docked_poses/pose_001.mol2
      
  - protein:
      id: ["A"]
      sequence: MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV
      structure_path: receptor.pdb
    ligand:
      id: ["B"]
      smiles: c1ccc(cc1)O
      structure_path: ./docked_poses/pose_002.mol2
```

### Individual YAML Output (one per pose)

```
yaml_configs/
├── pose_0000_pose_001.yaml
├── pose_0001_pose_002.yaml
└── pose_0002_pose_003.yaml
```

Each contains a single sequence.

## Validation

The script validates each pose:

- ✓ File exists
- ✓ Can be loaded as molecule
- ✓ Has 3D coordinates (conformer)
- ✓ Correct file format

Invalid poses are skipped with warnings. Use `--no-validate` to skip checks (not recommended).

## Troubleshooting

### Error: "No pose files matching pattern found"

Check:
1. Pattern matches your files: `ls poses_dir/pattern`
2. Change pattern: `--pattern "*.pdb"` or `--pattern "*.sdf"`
3. Verify directory path

### Error: "Could not load molecule"

The pose file might:
1. Have invalid format
2. Be corrupted
3. Be in wrong format for pattern

Try opening in RDKit or visualization tool to diagnose.

### Error: "No 3D coordinates"

The pose file needs embedded 3D coordinates:
- Vina outputs `.pdbqt` → convert to `.mol2`/`.sdf`/`.pdb`
- Use `obabel` to convert

```bash
obabel input.pdbqt -O output.mol2
```

### Error: "Sequence length mismatch"

The protein sequence doesn't match structure:
1. Check FASTA sequence is correct
2. Verify it matches the PDB chain
3. Use exact sequence from structure file

## Performance Tips

### For many poses (100+)

Use individual YAML files + parallel processing:

```bash
# Generate individual YAMLs
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output-dir ./yamls/

# Process with job scheduler (SLURM)
sbatch --array=1-$(ls yamls/*.yaml | wc -l) \
    run_affinity.sh

# Or with parallel on multiple cores
parallel -j $(nproc) python -m boltz.main_simplified predict \
    --yaml {} --checkpoint checkpoint.ckpt --output results/{/.}/ \
    ::: yamls/*.yaml
```

### For large proteins

Consider:
1. Using pocket constraints to focus affinity calculation
2. Running on GPU: `--device cuda`
3. Batch processing with reduced precision

## Integration with other tools

### Glide (Schrodinger)

```bash
# Export Glide poses as PDB
# Then convert to MOL2
obabel pose*.pdb -O pose*.mol2 -m

python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./ \
    --pattern "pose*.mol2" \
    --output glide_scoring.yaml
```

### SMINA

```bash
# Run SMINA
smina -r receptor.pdbqt -l ligand.pdb -o docked.sdf -n 20

# Convert to MOL2
obabel docked.sdf -O pose*.mol2 -m

python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --poses-dir ./ \
    --auto-smiles \
    --pattern "pose*.mol2" \
    --output smina_scoring.yaml
```

## API Usage (in Python)

```python
from scripts.generate_affinity_yaml import generate_batch_yaml, generate_individual_yamls
from pathlib import Path

# Single YAML for all poses
generate_batch_yaml(
    protein_pdb=Path("receptor.pdb"),
    protein_seq="MKFLKF...",
    ligand_smiles="c1ccccc1",
    poses_dir=Path("poses/"),
    output_yaml=Path("batch.yaml"),
    file_pattern="*.mol2",
)

# Individual YAML per pose
generate_individual_yamls(
    protein_pdb=Path("receptor.pdb"),
    protein_seq="MKFLKF...",
    ligand_smiles="c1ccccc1",
    poses_dir=Path("poses/"),
    output_dir=Path("yamls/"),
    file_pattern="*.mol2",
)
```
