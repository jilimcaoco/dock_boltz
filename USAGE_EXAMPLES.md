# Usage Examples - Affinity Prediction with External Structures

## Table of Contents
1. [Quick Start (CLI)](#quick-start-cli)
2. [Python API Examples](#python-api-examples)
3. [Advanced Use Cases](#advanced-use-cases)
4. [Preparing Your Own Data](#preparing-your-own-data)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start (CLI)

### Example 1: Single Prediction with External Structures (NEW!)

```bash
# Predict affinity using your own 3D structures from docking/modeling
python -m boltz.main_simplified predict \
    --input examples/affinity_with_structures.yaml \
    --output results.csv \
    --checkpoint /path/to/boltz2_affinity.ckpt \
    --device cuda
```

**Input** (`examples/affinity_with_structures.yaml`):
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ...
      structure_path: /path/to/protein.pdb  # Your external structure!
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
      structure_path: /path/to/ligand.mol2  # Your docked pose!
properties:
  - affinity:
      binder: B
```

**Supported Structure Formats:**
- **Proteins**: `.pdb`, `.cif`, `.mmcif`
- **Ligands**: `.mol2`, `.sdf`, `.pdb`

**Output** (`results.csv`):
```csv
complex_id,affinity_pred_value,affinity_probability_binary
complex_0_0,-6.2341,0.8234
```

---

### Example 1b: Single Prediction (Basic - No External Structures)

```bash
# Predict affinity (coordinates generated from SMILES)
python -m boltz.main_simplified predict \
    --input examples/affinity.yaml \
    --output results.csv \
    --checkpoint /path/to/boltz2_affinity.ckpt \
    --device cuda
```

**Input** (`examples/affinity.yaml`):
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ...
      # No structure_path = coords generated from sequence
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
      # No structure_path = coords generated from SMILES
properties:
  - affinity:
      binder: B
```

**Output** (`results.csv`):
```csv
complex_id,affinity_pred_value,affinity_probability_binary
complex_0_0,-6.2341,0.8234
```

**Interpretation:**
- `affinity_pred_value = -6.23` → IC50 ≈ 0.59 µM (10^-6.23)
- `affinity_probability_binary = 0.82` → 82% confidence it's a binder

---

### Example 2: Batch Prediction with Mixed Sources

```bash
# Create configs mixing external structures and generated coords
mkdir my_complexes

# Complex 1: Use your docked pose
cat > my_complexes/complex1.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNV...
      structure_path: docking/complex1_protein.pdb
  - ligand:
      id: B
      smiles: 'CCO'
      structure_path: docking/complex1_ligand.mol2
properties:
  - affinity:
      binder: B
EOF

# Complex 2: Generate from SMILES
cat > my_complexes/complex2.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNV...
  - ligand:
      id: B
      smiles: 'CC(C)O'
properties:
  - affinity:
      binder: B
EOF

# Run batch
python -m boltz.main_simplified predict \
    --input my_complexes/ \
    --output batch_results.csv \
    --checkpoint /path/to/boltz2_affinity.ckpt
```

---

### Example 2 (Original): Batch Prediction (Multiple Complexes)

```bash
# Create a directory with multiple YAML configs
mkdir my_complexes
cp complex1.yaml my_complexes/
cp complex2.yaml my_complexes/
cp complex3.yaml my_complexes/

# Run batch prediction
python -m boltz.main_simplified predict \
    --input my_complexes/ \
    --output batch_results.csv \
    --checkpoint /path/to/boltz2_affinity.ckpt \
    --device cuda \
    --recycling-steps 2
```

**Output** (`batch_results.csv`):
```csv
complex_id,affinity_pred_value,affinity_probability_binary
complex1_0,-7.4521,0.9123
complex2_0,-5.2341,0.6789
complex3_0,-4.1234,0.4521
```

---

### Example 3: CPU Inference (No GPU)

```bash
python -m boltz.main_simplified predict \
    --input examples/affinity.yaml \
    --output results_cpu.csv \
    --checkpoint /path/to/boltz2_affinity.ckpt \
    --device cpu
```

**Note:** CPU inference is ~10-20x slower but useful for debugging or small-scale predictions.

---

### Example 4: With Recycling (Higher Accuracy)

```bash
# Use 3 recycling steps for better evoformer embeddings
python -m boltz.main_simplified predict \
    --input examples/affinity.yaml \
    --output results_recycled.csv \
    --checkpoint /path/to/boltz2_affinity.ckpt \
    --device cuda \
    --recycling-steps 3
```

**Trade-off:** Each recycling step runs the pairformer again (~2x slower per step), but may improve accuracy.

---

## Python API Examples

### Example 1: Basic Python Script

```python
"""
basic_affinity_prediction.py

Predict affinity for a single protein-ligand complex.
"""

import torch
from pathlib import Path
from boltz.model.models.affinity_predictor import AffinityPredictor
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule

# Configuration
CHECKPOINT = "checkpoints/boltz2_affinity.ckpt"
INPUT_YAML = "examples/affinity.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT}...")
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
hparams = checkpoint["hyper_parameters"]

# Initialize model
print("Initializing model...")
model = AffinityPredictor(
    token_s=hparams.get("token_s", 384),
    token_z=hparams.get("token_z", 128),
    embedder_args=hparams.get("embedder_args", {}),
    msa_args=hparams.get("msa_args", {}),
    pairformer_args=hparams.get("pairformer_args", {}),
    affinity_model_args=hparams.get("affinity_model_args", {}),
    affinity_ensemble=hparams.get("affinity_ensemble", False),
    affinity_mw_correction=hparams.get("affinity_mw_correction", True),
)
model.load_state_dict(checkpoint["state_dict"], strict=False)
model = model.to(DEVICE)
model.eval()

# Load data
print(f"Loading data from {INPUT_YAML}...")
data_module = Boltz2InferenceDataModule(
    input_dir=Path(INPUT_YAML),
    batch_size=1,
    num_workers=0,
)
data_module.setup("predict")
dataloader = data_module.predict_dataloader()

# Run prediction
print("Running prediction...")
with torch.no_grad():
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # Get coordinates
        coords = batch["coords"]
        if coords.dim() == 4:
            coords = coords.squeeze(1)  # Remove conformer dimension
        
        # Predict
        predictions = model(
            feats=batch,
            coords=coords,
            recycling_steps=0,
        )
        
        # Extract results
        affinity_value = predictions["affinity_pred_value"].item()
        affinity_prob = predictions["affinity_probability_binary"].item()
        
        # Convert to IC50
        ic50_um = 10 ** affinity_value
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Affinity (log10 IC50): {affinity_value:.3f}")
        print(f"IC50 (µM):             {ic50_um:.3e}")
        print(f"Binder probability:    {affinity_prob:.1%}")
        print("="*60)
```

**Output:**
```
Loading checkpoint from checkpoints/boltz2_affinity.ckpt...
Initializing model...
Loading data from examples/affinity.yaml...
Running prediction...

============================================================
RESULTS
============================================================
Affinity (log10 IC50): -6.234
IC50 (µM):             5.828e-07
Binder probability:    82.3%
============================================================
```

---

### Example 2: Batch Processing with Custom Features

```python
"""
batch_affinity_with_external_structures.py

Process multiple protein-ligand complexes with external 3D structures.
"""

import torch
import pandas as pd
from pathlib import Path
from boltz.model.models.affinity_predictor import AffinityPredictor

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the affinity predictor model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    
    model = AffinityPredictor(
        token_s=hparams.get("token_s", 384),
        token_z=hparams.get("token_z", 128),
        embedder_args=hparams.get("embedder_args", {}),
        msa_args=hparams.get("msa_args", {}),
        pairformer_args=hparams.get("pairformer_args", {}),
        affinity_model_args=hparams.get("affinity_model_args", {}),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model

def predict_affinity(model, feats, coords, device="cuda"):
    """Run affinity prediction on prepared features."""
    # Move to device
    feats = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in feats.items()}
    coords = coords.to(device)
    
    with torch.no_grad():
        predictions = model(
            feats=feats,
            coords=coords,
            recycling_steps=0,
        )
    
    return {
        "affinity_value": predictions["affinity_pred_value"].cpu().item(),
        "affinity_probability": predictions["affinity_probability_binary"].cpu().item(),
        "ic50_um": 10 ** predictions["affinity_pred_value"].cpu().item(),
    }

# Main script
if __name__ == "__main__":
    CHECKPOINT = "checkpoints/boltz2_affinity.ckpt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model once
    print("Loading model...")
    model = load_model(CHECKPOINT, DEVICE)
    
    # Your prepared data (from external source)
    complexes = [
        {
            "name": "compound_A_vs_target1",
            "feats": ...,  # Prepared features dict
            "coords": ..., # Your 3D structure (N_atoms, 3)
        },
        {
            "name": "compound_B_vs_target1",
            "feats": ...,
            "coords": ...,
        },
        # ... more complexes
    ]
    
    # Batch predict
    results = []
    for i, complex_data in enumerate(complexes):
        print(f"Processing {complex_data['name']} ({i+1}/{len(complexes)})...")
        
        pred = predict_affinity(
            model,
            complex_data["feats"],
            complex_data["coords"],
            DEVICE
        )
        
        results.append({
            "complex_name": complex_data["name"],
            "affinity_log10_ic50": pred["affinity_value"],
            "ic50_um": pred["ic50_um"],
            "binder_probability": pred["affinity_probability"],
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("batch_predictions.csv", index=False)
    print(f"\nResults saved to batch_predictions.csv")
    print(df.to_string())
```

**Output:**
```
Loading model...
Processing compound_A_vs_target1 (1/2)...
Processing compound_B_vs_target1 (2/2)...

Results saved to batch_predictions.csv
               complex_name  affinity_log10_ic50    ic50_um  binder_probability
0  compound_A_vs_target1              -7.234  5.828e-08              0.9123
1  compound_B_vs_target1              -5.123  7.534e-06              0.6789
```

---

### Example 3: Integration with RDKit/Structure Prediction

```python
"""
integrate_with_external_docking.py

Use affinity predictor with structures from external docking tools.
"""

import torch
from rdkit import Chem
from boltz.model.models.affinity_predictor import AffinityPredictor
from boltz.data.feature.featurizerv2 import FeaturizerV2

def predict_from_pdb_and_smiles(
    protein_pdb: str,
    ligand_smiles: str,
    checkpoint_path: str,
    chain_id: str = "B"
):
    """
    Predict affinity from a protein PDB file and ligand SMILES.
    
    Parameters
    ----------
    protein_pdb : str
        Path to protein PDB file
    ligand_smiles : str
        SMILES string for ligand
    checkpoint_path : str
        Path to model checkpoint
    chain_id : str
        Chain ID to assign to ligand
    
    Returns
    -------
    dict
        Affinity predictions
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AffinityPredictor(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device).eval()
    
    # Prepare features (you need to implement this based on your data format)
    # This is a placeholder - actual implementation depends on your featurizer
    featurizer = FeaturizerV2(...)
    feats = featurizer.featurize(
        protein_pdb=protein_pdb,
        ligand_smiles=ligand_smiles,
        ligand_chain_id=chain_id,
    )
    
    # Extract coordinates from PDB
    # coords = extract_coords_from_pdb(protein_pdb, ligand_smiles)
    coords = feats["coords"]  # Assuming featurizer includes coords
    
    # Predict
    with torch.no_grad():
        predictions = model(
            feats={k: v.to(device) for k, v in feats.items()},
            coords=coords.to(device),
            recycling_steps=0,
        )
    
    return {
        "affinity_log10_ic50": predictions["affinity_pred_value"].cpu().item(),
        "ic50_um": 10 ** predictions["affinity_pred_value"].cpu().item(),
        "binder_probability": predictions["affinity_probability_binary"].cpu().item(),
    }

# Example usage
if __name__ == "__main__":
    result = predict_from_pdb_and_smiles(
        protein_pdb="my_protein.pdb",
        ligand_smiles="CCO",  # Ethanol
        checkpoint_path="checkpoints/boltz2_affinity.ckpt",
        chain_id="B"
    )
    
    print(f"Predicted IC50: {result['ic50_um']:.2e} µM")
    print(f"Binder probability: {result['binder_probability']:.1%}")
```

---

## Advanced Use Cases

### Example 1: Virtual Screening Pipeline

```python
"""
virtual_screening.py

Screen a library of compounds against a target protein.
"""

import torch
import pandas as pd
from tqdm import tqdm
from boltz.model.models.affinity_predictor import AffinityPredictor

# Load model once
checkpoint = torch.load("checkpoints/boltz2_affinity.ckpt", weights_only=False)
model = AffinityPredictor(**checkpoint["hyper_parameters"])
model.load_state_dict(checkpoint["state_dict"], strict=False)
model = model.cuda().eval()

# Load compound library
compound_library = pd.read_csv("compound_library.csv")
# Columns: compound_id, smiles

# Target protein
target_protein_seq = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ..."

# Screen all compounds
results = []
for idx, row in tqdm(compound_library.iterrows(), total=len(compound_library)):
    # Prepare complex (protein + compound)
    # feats, coords = prepare_complex(target_protein_seq, row['smiles'])
    
    # Predict affinity
    with torch.no_grad():
        pred = model(feats, coords, recycling_steps=0)
    
    results.append({
        "compound_id": row["compound_id"],
        "smiles": row["smiles"],
        "affinity": pred["affinity_pred_value"].item(),
        "probability": pred["affinity_probability_binary"].item(),
        "ic50_um": 10 ** pred["affinity_pred_value"].item(),
    })

# Save ranked results
df = pd.DataFrame(results)
df = df.sort_values("affinity", ascending=True)  # Lower is better
df.to_csv("virtual_screening_results.csv", index=False)

# Top 10 hits
print("\nTop 10 Predicted Binders:")
print(df.head(10))
```

---

### Example 2: Ensemble Predictions (Multiple Structures)

```python
"""
ensemble_prediction.py

Average predictions over multiple conformations.
"""

import torch
import numpy as np
from boltz.model.models.affinity_predictor import AffinityPredictor

def ensemble_predict(model, feats, coords_list, device="cuda"):
    """
    Predict affinity averaging over multiple structures.
    
    Parameters
    ----------
    model : AffinityPredictor
        Loaded model
    feats : dict
        Feature dictionary
    coords_list : list of torch.Tensor
        List of coordinate tensors, each (N_atoms, 3)
    device : str
        Device to use
    
    Returns
    -------
    dict
        Averaged predictions
    """
    predictions = []
    
    with torch.no_grad():
        for coords in coords_list:
            pred = model(
                feats={k: v.to(device) for k, v in feats.items()},
                coords=coords.to(device),
                recycling_steps=0,
            )
            predictions.append({
                "affinity": pred["affinity_pred_value"].cpu().item(),
                "probability": pred["affinity_probability_binary"].cpu().item(),
            })
    
    # Average predictions
    avg_affinity = np.mean([p["affinity"] for p in predictions])
    avg_probability = np.mean([p["probability"] for p in predictions])
    std_affinity = np.std([p["affinity"] for p in predictions])
    
    return {
        "affinity_mean": avg_affinity,
        "affinity_std": std_affinity,
        "probability_mean": avg_probability,
        "ic50_um": 10 ** avg_affinity,
    }

# Example: predict over 10 docked poses
model = ...  # Load model
feats = ...  # Prepare features
docked_poses = [...]  # List of 10 coordinate tensors

result = ensemble_predict(model, feats, docked_poses)
print(f"Average IC50: {result['ic50_um']:.2e} ± {result['affinity_std']:.2f} log units")
```

---

## Preparing Your Own Data

### Using Structures from Docking Tools

#### Example: AutoDock Vina Output

```bash
# After running AutoDock Vina, you have:
# - receptor.pdbqt
# - ligand_out.pdbqt

# Convert to supported formats
obabel receptor.pdbqt -O receptor.pdb
obabel ligand_out.pdbqt -O ligand.mol2

# Create config
cat > vina_complex.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ...  # Your protein sequence
      structure_path: receptor.pdb
  - ligand:
      id: B
      smiles: 'CCO'  # Your ligand SMILES
      structure_path: ligand.mol2  # Docked pose
properties:
  - affinity:
      binder: B
EOF

# Predict affinity
python -m boltz.main_simplified predict \
    --input vina_complex.yaml \
    --output vina_affinity.csv \
    --checkpoint model.ckpt
```

#### Example: Glide/Schrödinger Output

```bash
# Glide outputs Maestro files (.mae)
# Convert to PDB/MOL2
$SCHRODINGER/utilities/structconvert ligand_pose.mae ligand_pose.mol2

# Create config with external structure
cat > glide_complex.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_SEQUENCE_HERE
      structure_path: protein.pdb
  - ligand:
      id: B
      smiles: 'YOUR_SMILES_HERE'
      structure_path: ligand_pose.mol2
properties:
  - affinity:
      binder: B
EOF
```

#### Example: AlphaFold3 Structures

```bash
# If you have AlphaFold3 predictions with ligands
cat > af3_complex.yaml << EOF
version: 1
sequences:
  - protein:
      id: A
      sequence: YOUR_SEQUENCE
      structure_path: af3_prediction.pdb  # AF3 output
  - ligand:
      id: B  
      smiles: 'YOUR_SMILES'
      # Extract ligand coords from AF3 PDB or use separate file
      structure_path: ligand_from_af3.sdf
properties:
  - affinity:
      binder: B
EOF
```

### Creating a YAML Config from Your Data

```python
"""
create_config_from_data.py

Helper script to create YAML configs from your protein/ligand data.
"""

import yaml
from pathlib import Path

def create_affinity_config(
    protein_sequence: str,
    ligand_smiles: str,
    protein_id: str = "A",
    ligand_id: str = "B",
    output_path: str = "my_config.yaml"
):
    """Create a Boltz affinity config YAML file."""
    
    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": protein_id,
                    "sequence": protein_sequence,
                }
            },
            {
                "ligand": {
                    "id": ligand_id,
                    "smiles": ligand_smiles,
                }
            }
        ],
        "properties": [
            {
                "affinity": {
                    "binder": ligand_id
                }
            }
        ]
    }
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to {output_path}")

# Example usage
if __name__ == "__main__":
    protein_seq = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ..."  # Your protein
    ligand_smi = "CCO"  # Your ligand (ethanol)
    
    create_affinity_config(
        protein_sequence=protein_seq,
        ligand_smiles=ligand_smi,
        output_path="my_complex.yaml"
    )
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Checkpoint File
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'model.ckpt'
```

**Solution:**
```bash
# Download the checkpoint (placeholder - use actual download link)
wget https://your-model-host.com/boltz2_affinity.ckpt -O checkpoints/model.ckpt

# Or specify correct path
python -m boltz.main_simplified predict \
    --checkpoint /correct/path/to/model.ckpt \
    ...
```

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU
python -m boltz.main_simplified predict \
    --device cpu \
    ...

# Or reduce batch size
python -m boltz.main_simplified predict \
    --batch-size 1 \
    ...
```

#### 3. Missing Coordinates in Input
```
KeyError: 'coords'
```

**Solution:**
Ensure your YAML config or feature dict includes 3D coordinates. The model expects pre-computed structures.

#### 4. Incompatible Checkpoint Format
```
RuntimeError: Error(s) in loading state_dict
```

**Solution:**
```python
# Use strict=False to allow partial loading
model.load_state_dict(checkpoint["state_dict"], strict=False)
```

---

## Performance Tips

1. **Use GPU**: ~10-20x faster than CPU
2. **Batch processing**: Process multiple complexes together
3. **Minimize recycling**: Start with 0 recycling steps, increase only if needed
4. **Cache features**: Featurization is expensive - cache prepared features
5. **Half precision**: Use `.half()` for faster inference (may reduce accuracy slightly)

```python
# Example: half precision inference
model = model.half()
coords = coords.half()
# Run prediction...
```

---

## Summary

The simplified API now supports:
- ✅ Command-line batch predictions
- ✅ Python API for integration
- ✅ External structure input
- ✅ Ensemble predictions
- ✅ Virtual screening workflows

All focused on affinity prediction without the overhead of structure generation!
