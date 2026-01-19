# Boltz-2 Affinity Batch Scoring Toolkit

Complete tooling for scoring docked poses with Boltz-2 affinity module.

## What You Can Do

✅ **Score docked poses** - From any docking tool (Vina, Glide, SMINA, etc.)  
✅ **Batch processing** - Score 100+ poses efficiently  
✅ **Parallel execution** - Leverage multiple GPUs/CPUs  
✅ **Auto-detect ligands** - Extract SMILES from structure files  
✅ **Compare with docking** - Rank poses by Boltz affinity  

## Tools

### 1. YAML Generator
**Purpose:** Generate configuration files for batch scoring  
**Script:** `scripts/generate_affinity_yaml.py`  
**Use when:** You have docked poses and want to score them  

```bash
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir ./poses/ \
    --output batch_affinity.yaml
```

**Quick reference:** [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md)

### 2. Affinity Predictor
**Purpose:** Score structures using Boltz-2 affinity module  
**Script:** `python -m boltz.main_simplified predict`  
**Use when:** You have YAML configs ready  

```bash
python -m boltz.main_simplified predict \
    --yaml batch_affinity.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/
```

**Reference:** Run `python -m boltz.main_simplified predict --help`

### 3. Test Suite
**Purpose:** Validate external structure loading  
**Script:** `test_external_structures.py`  
**Use when:** Setting up the pipeline for the first time  

```bash
python test_external_structures.py
```

## Quick Start (5 minutes)

### For Vina users:

```bash
# 1. Run Vina (you probably already have this)
vina --receptor target.pdbqt --ligand ligand.pdbqt --out docked.pdbqt --num_modes 20

# 2. Convert poses to MOL2
mkdir -p poses
obabel docked.pdbqt -O poses/pose_%04d.mol2 -m

# 3. Generate YAML
python scripts/generate_affinity_yaml.py \
    --protein target.pdb \
    --protein-seq "$(tail -1 target.fasta)" \
    --ligand "c1ccccc1" \
    --poses-dir poses/ \
    --output affinity_batch.yaml

# 4. Score with Boltz
python -m boltz.main_simplified predict \
    --yaml affinity_batch.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/

# 5. Check results
ls -lh results/
```

## Example Workflows

### Workflow 1: Single ligand, multiple poses

```bash
# Generate YAML for all poses
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir poses/ \
    --output batch.yaml

# Score all in single run
python -m boltz.main_simplified predict \
    --yaml batch.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/
```

### Workflow 2: Parallel processing

```bash
# Generate individual YAML per pose
python scripts/generate_affinity_yaml.py \
    --protein receptor.pdb \
    --protein-seq "MKFLKF..." \
    --ligand "c1ccccc1" \
    --poses-dir poses/ \
    --output-dir yaml_configs/

# Process in parallel (4 jobs)
ls yaml_configs/*.yaml | xargs -P 4 -I {} \
    python -m boltz.main_simplified predict \
        --yaml {} \
        --checkpoint checkpoint.ckpt \
        --output results/{/.}
```

### Workflow 3: Multiple ligands

```bash
for ligand_smi in "c1ccccc1" "c1ccc(O)cc1" "c1cccnc1"; do
    python scripts/generate_affinity_yaml.py \
        --protein receptor.pdb \
        --protein-seq "MKFLKF..." \
        --ligand "$ligand_smi" \
        --poses-dir "poses_ligand/" \
        --output "batch_${ligand_smi}.yaml"
done

# Score each
for yaml in batch_*.yaml; do
    python -m boltz.main_simplified predict \
        --yaml "$yaml" \
        --checkpoint checkpoint.ckpt \
        --output "results_${yaml%.yaml}/"
done
```

## File Formats Supported

| Type | Formats |
|------|---------|
| Protein | PDB, mmCIF |
| Ligand/Poses | MOL2, SDF, PDB |

**Requirement:** All files must have embedded 3D coordinates.

## Installation

```bash
# Core dependencies
pip install torch pyyaml click rdkit-pypi

# Optional: For post-processing
pip install pandas numpy

# Optional: For structure conversion
brew install open-babel  # macOS
apt-get install openbabel  # Linux
conda install -c conda-forge openbabel  # Conda
```

## Documentation

| Document | Purpose |
|----------|---------|
| [YAML_GENERATOR_QUICK_REF.md](YAML_GENERATOR_QUICK_REF.md) | Quick reference for YAML generator |
| [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md) | Comprehensive batch scoring guide |
| [EXTERNAL_STRUCTURES_GUIDE.md](EXTERNAL_STRUCTURES_GUIDE.md) | How external structures work |
| [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) | Testing and validation guide |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | One-page reference |
| [examples/batch_scoring_example/README.md](examples/batch_scoring_example/README.md) | Full working example |

## Common Tasks

### Task: Extract top-10 poses from Boltz results

```python
import pandas as pd
import yaml

# Load Boltz predictions
with open("results/predictions.yaml") as f:
    predictions = yaml.safe_load(f)

# Create dataframe
df = pd.DataFrame([
    {"pose": p["name"], "affinity": p["metrics"]["affinity"]}
    for p in predictions
])

# Get top 10
top10 = df.nsmallest(10, "affinity")
print(top10)
top10.to_csv("top10_poses.csv", index=False)
```

### Task: Compare Vina vs Boltz scores

```python
import pandas as pd
import yaml

# Extract from Vina
vina_scores = {}
with open("docked.pdbqt") as f:
    for line in f:
        if line.startswith("RESULT"):
            pose_num = len(vina_scores) + 1
            vina_scores[f"pose_{pose_num:04d}"] = float(line.split()[1])

# Load Boltz
with open("results/predictions.yaml") as f:
    boltz = yaml.safe_load(f)

# Compare
df = pd.DataFrame([
    {"pose": p["name"], "boltz": p["metrics"]["affinity"], 
     "vina": vina_scores.get(p["name"].split("/")[-1].replace(".mol2", ""), None)}
    for p in boltz
])

print(df.corr())  # Correlation between scores
df.to_csv("comparison.csv", index=False)
```

### Task: Filter top Vina poses before Boltz scoring

```bash
# Extract Vina top-10
python << 'EOF'
vina_scores = []
with open("docked.pdbqt") as f:
    for line in f:
        if line.startswith("RESULT"):
            vina_scores.append(float(line.split()[1]))

top_indices = sorted(range(len(vina_scores)), key=lambda i: vina_scores[i])[:10]
print(f"Top 10 indices: {top_indices}")
EOF

# Create YAML only for top poses
# (Modify generate_affinity_yaml.py or manually select files)
```

## Troubleshooting

**Q: How do I extract SMILES from my ligand?**
```bash
# Option 1: Use SMILES from paper/database
# Option 2: Auto-detect from structure
python scripts/generate_affinity_yaml.py \
    --auto-smiles  # Extracts from first pose
# Option 3: Use RDKit
python -c "from rdkit import Chem; print(Chem.MolToSmiles(Chem.SDMolSupplier('ligand.sdf')[0]))"
```

**Q: My poses are in PDB format, what do I do?**
```bash
# Use --pattern "*.pdb" 
python scripts/generate_affinity_yaml.py \
    --pattern "*.pdb" \
    ...
```

**Q: Can I score poses from Glide/SMINA/other tools?**
```bash
# Yes! Convert to MOL2/SDF/PDB first
obabel glide_poses.pdb -O pose_%04d.mol2 -m

# Then use YAML generator
```

**Q: How do I run on GPU?**
```bash
python -m boltz.main_simplified predict \
    --yaml batch.yaml \
    --checkpoint checkpoint.ckpt \
    --output results/ \
    --device cuda  # or cuda:0, cuda:1, etc.
```

**Q: What's the typical scoring speed?**
- CPU: 1-5 poses/minute (depends on protein size)
- GPU: 10-50 poses/minute (depends on GPU memory)
- Use parallel processing for batches

## Support

For issues:
1. Check [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md) troubleshooting section
2. Check [BATCH_AFFINITY_SCORING.md](BATCH_AFFINITY_SCORING.md) FAQ
3. Verify structure files with `test_external_structures.py`
4. Check log files in output directory

## Citation

If you use this toolkit, please cite:

- **Boltz-2:** Lindorff-Larsen et al. (2024) - Coming soon
- **AutoDock Vina:** Trott & Olson (2010)
- **RDKit:** Landrum et al. (https://www.rdkit.org/)

## Next Steps

1. **Install dependencies** - See Installation section
2. **Test with example** - See examples/batch_scoring_example/
3. **Run YAML generator** - See YAML_GENERATOR_QUICK_REF.md
4. **Score poses** - See Affinity Predictor section
5. **Post-process results** - See Common Tasks section

Happy scoring! 🎯
