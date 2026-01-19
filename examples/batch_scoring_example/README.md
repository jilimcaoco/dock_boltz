# Example: Batch Pose Scoring Workflow

This directory structure shows a typical workflow for docking + Boltz-2 affinity scoring.

## Directory Layout

```
project/
├── target.pdb                    # Protein structure (receptor)
├── target.fasta                  # Protein sequence
├── ligand.smi                    # Ligand SMILES
├── docking/
│   ├── target.pdbqt             # Prepared protein for Vina
│   ├── ligand.pdbqt             # Prepared ligand for Vina
│   └── docked.pdbqt             # Vina output (all poses)
├── poses/                        # Converted poses
│   ├── pose_001.mol2
│   ├── pose_002.mol2
│   ├── pose_003.mol2
│   └── pose_010.mol2
├── scripts/
│   └── generate_affinity_yaml.py  # YAML generator
├── batch_affinity.yaml           # Generated YAML (all poses)
├── affinity_results/             # Predictions output
│   ├── log.txt
│   ├── predictions.yaml
│   └── scores.csv
└── README.md                     # This workflow
```

## Step-by-Step Workflow

### Step 1: Prepare structures

```bash
# Extract protein sequence
grep "^>" target.pdb || python -c "
from Bio import PDB
p = PDB.PDBParser()
s = p.get_structure('target', 'target.pdb')
ppb = PDB.PPBuilder()
for pp in ppb.build_peptides(s):
    print(pp.get_sequence())
" > target.fasta

# Verify SMILES for your ligand
echo "c1ccc(cc1)O" > ligand.smi
```

### Step 2: Run molecular docking (Vina)

```bash
# Install AutoDock Vina
# brew install autodock-vina  # macOS
# apt-get install autodock-vina  # Linux
# conda install -c conda-forge autodock-vina  # Conda

# Prepare PDBQT files (requires meeko or AutoDockTools)
meeko_prepare_ligand.py -i ligand.pdb -o docking/ligand.pdbqt
meeko_prepare_receptor.py -i target.pdb -o docking/target.pdbqt

# Run docking (20 poses)
cd docking/
vina \
    --receptor target.pdbqt \
    --ligand ligand.pdbqt \
    --center_x 10.0 \
    --center_y 20.0 \
    --center_z 30.0 \
    --size_x 20 \
    --size_y 20 \
    --size_z 20 \
    --out docked.pdbqt \
    --num_modes 20

cd ..
```

### Step 3: Convert poses to MOL2

```bash
# Install OpenBabel
# brew install open-babel  # macOS
# apt-get install openbabel  # Linux
# conda install -c conda-forge openbabel  # Conda

# Create poses directory
mkdir -p poses

# Split docked.pdbqt into individual poses
obabel docking/docked.pdbqt -O poses/pose_%04d.mol2 -m
```

### Step 4: Generate YAML configs

```bash
python scripts/generate_affinity_yaml.py \
    --protein target.pdb \
    --protein-seq "$(cat target.fasta | tail -1)" \
    --ligand "$(cat ligand.smi)" \
    --poses-dir poses/ \
    --pattern "pose_*.mol2" \
    --output batch_affinity.yaml

# Check generated YAML
head -30 batch_affinity.yaml
wc -l batch_affinity.yaml  # Should have ~20 sequences
```

### Step 5: Download Boltz-2 checkpoint (first time)

```bash
# Download checkpoint from https://huggingface.co/boltz-community/Boltz-2
# Or use pre-downloaded checkpoint at:
export CHECKPOINT="/path/to/boltz2_affinity_checkpoint.ckpt"
```

### Step 6: Run affinity predictions

```bash
mkdir -p affinity_results

python -m boltz.main_simplified predict \
    --yaml batch_affinity.yaml \
    --checkpoint "${CHECKPOINT}" \
    --output affinity_results/ \
    --device cuda  # Add if you have GPU

# Check results
ls -lh affinity_results/
cat affinity_results/log.txt | tail -20
```

### Step 7: Extract and rank scores

```bash
# Parse predictions and rank by affinity
python << 'EOF'
import yaml
import pandas as pd
from pathlib import Path

# Load results
results_dir = Path("affinity_results")
predictions = yaml.safe_load(open(results_dir / "predictions.yaml"))

# Extract scores
data = []
for pred in predictions:
    # Each prediction includes affinity score
    # Structure: {"name": "...", "metrics": {"affinity": -7.5}, ...}
    data.append({
        "pose": pred.get("name", "unknown"),
        "affinity": pred.get("metrics", {}).get("affinity", float("nan")),
    })

# Create dataframe and rank
df = pd.DataFrame(data)
df = df.sort_values("affinity")  # Best binders have lowest affinity

# Save results
df.to_csv("affinity_results/scores.csv", index=False)
print(df)
print(f"\nBest pose: {df.iloc[0]['pose']} (Affinity: {df.iloc[0]['affinity']:.2f})")
EOF
```

### Step 8: Compare with docking scores (optional)

```bash
# Extract Vina scores from docked.pdbqt
python << 'EOF'
import pandas as pd

vina_scores = []
with open("docking/docked.pdbqt") as f:
    for i, line in enumerate(f):
        if line.startswith("RESULT"):
            parts = line.split()
            affinity = float(parts[1])
            vina_scores.append({
                "pose": f"pose_{i:04d}",
                "vina_affinity": affinity,
            })

df_vina = pd.DataFrame(vina_scores)

# Load Boltz scores
df_boltz = pd.read_csv("affinity_results/scores.csv")

# Merge
df_comparison = df_boltz.merge(df_vina, on="pose")
df_comparison = df_comparison.sort_values("affinity")

# Save comparison
df_comparison.to_csv("affinity_results/comparison.csv", index=False)
print(df_comparison)

# Calculate correlation
corr = df_comparison["affinity"].corr(df_comparison["vina_affinity"])
print(f"\nCorrelation with Vina: {corr:.3f}")
EOF
```

## Expected Output

### batch_affinity.yaml structure

```yaml
version: 2
sequences:
  - protein:
      id: ["A"]
      sequence: MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV
      structure_path: target.pdb
    ligand:
      id: ["B"]
      smiles: c1ccc(cc1)O
      structure_path: poses/pose_0001.mol2
      
  - protein:
      id: ["A"]
      sequence: MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV
      structure_path: target.pdb
    ligand:
      id: ["B"]
      smiles: c1ccc(cc1)O
      structure_path: poses/pose_0002.mol2

# ... more sequences ...
```

### Final scores (affinity_results/scores.csv)

```csv
pose,affinity
poses/pose_0005.mol2,-7.8
poses/pose_0002.mol2,-7.3
poses/pose_0008.mol2,-6.9
poses/pose_0001.mol2,-6.2
poses/pose_0003.mol2,-5.8
```

### Comparison with Vina (comparison.csv)

```csv
pose,affinity,vina_affinity
poses/pose_0005.mol2,-7.8,-8.1
poses/pose_0002.mol2,-7.3,-7.9
poses/pose_0008.mol2,-6.9,-7.2
poses/pose_0001.mol2,-6.2,-7.5
```

## Tips & Tricks

### Parallel GPU Processing

```bash
# If you have multiple GPUs, process in parallel
for pose in poses/pose_*.mol2; do
    gpu_id=$(($(basename $pose | cut -d_ -f2) % 4))
    CUDA_VISIBLE_DEVICES=$gpu_id python -m boltz.main_simplified predict \
        --yaml batch_affinity.yaml \
        --checkpoint checkpoint.ckpt \
        --output "affinity_results_gpu_${gpu_id}/" &
done
wait
```

### Large-Scale Screening

For 1000+ poses, use individual YAMLs + job scheduler:

```bash
# Generate individual YAMLs
python scripts/generate_affinity_yaml.py \
    --protein target.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir poses/ \
    --output-dir yaml_configs/

# Submit to SLURM
sbatch -a 1-$(ls yaml_configs/*.yaml | wc -l) run_array_job.sh
```

### Filter by Vina Score Before Boltz

```bash
# Only score top 10 Vina poses
python << 'EOF'
from pathlib import Path
import subprocess

# Extract Vina scores
vina_scores = {}
for line in open("docking/docked.pdbqt"):
    if line.startswith("RESULT"):
        pose_num = len(vina_scores) + 1
        affinity = float(line.split()[1])
        vina_scores[f"pose_{pose_num:04d}"] = affinity

# Get top 10
top_poses = sorted(vina_scores.items(), key=lambda x: x[1])[:10]
top_pose_names = [p[0] for p in top_poses]

# Generate YAML only for top poses
pose_files = sorted(Path("poses").glob("pose_*.mol2"))
filtered_poses = [p for p in pose_files if p.stem in top_pose_names]

# Generate YAML with filtered poses...
EOF
```

## Troubleshooting

### Issue: "No pose files matching pattern found"

```bash
# Check your pattern matches files
ls poses/  # What files exist?

# Update pattern
python scripts/generate_affinity_yaml.py \
    --pattern "*.mol2"  # Or "pose_*.mol2", "*.sdf", etc.
```

### Issue: "Could not load molecule"

```bash
# Validate pose files
python << 'EOF'
from rdkit import Chem
import os

for f in os.listdir("poses"):
    path = f"poses/{f}"
    try:
        if f.endswith(".mol2"):
            mol = Chem.MolFromMol2File(path, removeHs=False)
        else:
            mol = Chem.SDMolSupplier(path, removeHs=False)[0]
        
        if mol and mol.GetNumConformers() > 0:
            print(f"✓ {f}: {mol.GetNumAtoms()} atoms, 3D coords")
        else:
            print(f"✗ {f}: No 3D coords")
    except Exception as e:
        print(f"✗ {f}: {e}")
EOF

# If many fail, re-convert from PDBQT
obabel docking/docked.pdbqt -O poses/pose_%04d.mol2 -m
```

### Issue: Slow processing

```bash
# Use GPU
python -m boltz.main_simplified predict \
    --yaml batch_affinity.yaml \
    --checkpoint checkpoint.ckpt \
    --output affinity_results/ \
    --device cuda

# Or batch multiple poses
# See "Parallel GPU Processing" above
```

## Next Steps

1. **Validate predictions** - Compare Boltz scores with experimental data
2. **Docking refinement** - Use top Boltz-scored poses for MD
3. **Free energy** - Run MM-PBSA on best poses for more accurate ranking
4. **ML model** - Train ML model combining Vina + Boltz scores

See [BATCH_AFFINITY_SCORING.md](../BATCH_AFFINITY_SCORING.md) for more details.
