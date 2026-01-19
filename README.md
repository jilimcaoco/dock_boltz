# Boltz-2 Affinity Predictor

A simplified, streamlined version of Boltz-2 focused on **protein-ligand binding affinity prediction** using external 3D structures.

This is a cleaned-up fork of Boltz-2 that removes structure prediction (diffusion), confidence prediction, and training code, leaving only:
- **Evoformer** (sequence embedding + pairwise representation learning)
- **Affinity Module** (structure-aware binding prediction)

**Perfect for:** Screening compounds against a known protein target when you already have 3D structures from another source (docking, molecular dynamics, AlphaFold, etc.).

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/boltz-affinity.git
cd boltz-affinity
pip install -e .
```

### Basic Usage

```bash
# Validate your input config
python -m boltz.main_simplified validate --input affinity.yaml

# Run affinity prediction
python -m boltz.main_simplified predict \
  --input affinity.yaml \
  --output predictions.csv \
  --checkpoint boltz2_aff.ckpt \
  --device cuda
```

### Input Format

#### With External Structures (NEW!)

Provide your own 3D structures from docking, MD, or other sources:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ...
      structure_path: /path/to/protein.pdb  # Your structure!
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
      structure_path: /path/to/ligand.mol2  # Your docked pose!
properties:
  - affinity:
      binder: B
```

**Supported formats:**
- Proteins: `.pdb`, `.cif`, `.mmcif`
- Ligands: `.mol2`, `.sdf`, `.pdb`

#### Without External Structures

Generate coordinates automatically from sequences/SMILES:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
properties:
  - affinity:
      binder: B
```

### Output

Results are saved as CSV with columns:
- `complex_id`: Identifier for the complex
- `affinity_pred_value`: Predicted log10(IC50) in μM (lower = stronger binding)
- `affinity_probability_binary`: Predicted probability of being a binder (0-1)

## What's Included

### Core Components
- **InputEmbedder** (`src/boltz/model/modules/trunkv2.py`) - Converts sequences to token embeddings
- **MSAModule** - Multi-sequence alignment processing
- **Pairformer/Evoformer** - Token-level pairwise representation learning
- **AffinityModule** (`src/boltz/model/modules/affinity.py`) - Structure-aware affinity prediction

### Data Utilities
- Feature preparation pipeline (featurizerv2)
- Parsing utilities (FASTA, YAML, PDB)
- Token and atom-level representations

## What's Removed

To reduce code complexity and tech debt:
- ❌ Diffusion model (structure prediction)
- ❌ Confidence prediction module
- ❌ Training code and loss functions
- ❌ Evaluation scripts
- ❌ Template processing
- ❌ B-factor prediction
- ❌ Distogram head
- ❌ Old Boltz-1 implementation
- ❌ Complex optimizers and schedulers

## Architecture

```
Input Sequence + 3D Structure
        ↓
InputEmbedder (atom → token embeddings)
        ↓
MSAModule (sequence alignment integration)
        ↓
Pairformer/Evoformer (token-level embedding learning)
        ↓
AffinityModule (structure conditioning + prediction)
        ↓
affinity_pred_value + affinity_probability_binary
```

## Key Features

### Two Affinity Predictions
1. **`affinity_pred_value`** (log10 scale)
   - Predicted binding affinity in μM
   - Use for: Ligand optimization, SAR (structure-activity relationship)
   - Range: typically -2 to 7 (0.01 to 10,000 μM)

2. **`affinity_probability_binary`** (0-1 scale)
   - Probability of being a binder vs non-binder
   - Use for: Hit discovery, virtual screening, hit-to-lead triage
   - Higher values indicate likely binders

### Optional Molecular Weight Correction
When using ensemble mode, applies empirically-derived coefficients to account for ligand size bias:
```python
affinity_pred_value = 1.035 * raw_prediction - 0.600 * mw^0.3 + 2.833
```

## Model Capabilities

- **Protein target**: Any size (typically 50-5000 residues)
- **Ligand**: Small molecules up to 128 heavy atoms (trained on <56 atoms)
- **Structures**: Any 3D coordinates (PDB, docking predictions, MD frames, etc.)
- **Accuracy**: ~0.6 Pearson correlation with experimental IC50 on diverse benchmarks

## Python API

```python
import torch
from pathlib import Path
from boltz.model.models.affinity_predictor import AffinityPredictor
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule

# Load model
checkpoint = torch.load("boltz2_aff.ckpt")
model = AffinityPredictor(...)
model.load_state_dict(checkpoint["state_dict"])

# Prepare data
data_module = Boltz2InferenceDataModule(input_dir=Path("configs/"))
data_module.setup("predict")

# Run predictions
model.eval()
with torch.no_grad():
    for batch in data_module.predict_dataloader():
        coords = batch["coords"]
        predictions = model(feats=batch, coords=coords)
        affinity = predictions["affinity_pred_value"]
        binder_prob = predictions["affinity_probability_binary"]
```

## Configuration

The `AffinityPredictor` model accepts configuration for:
- Token embedding dimensions
- Evoformer architecture (number of blocks, heads)
- Affinity module settings
- Ensemble mode and molecular weight correction

See `src/boltz/model/models/affinity_predictor.py` for full parameters.

## Codebase Structure

```
src/boltz/
├── data/
│   ├── feature/featurizerv2.py      # Feature preparation
│   ├── parse/                        # Input parsing (FASTA, YAML, PDB)
│   ├── crop/affinity.py              # Spatial cropping
│   ├── tokenize/boltz2.py            # Tokenization
│   └── ...
├── model/
│   ├── modules/
│   │   ├── affinity.py               # ✓ Affinity prediction
│   │   ├── trunkv2.py                # ✓ Input embedder, pairwise init
│   │   ├── encodersv2.py             # ✓ Attention encoders
│   │   ├── transformersv2.py         # ✓ Token transformers
│   │   └── utils.py                  # ✓ Utilities
│   ├── layers/
│   │   ├── pairformer.py             # ✓ Pairformer/evoformer
│   │   └── ...                       # ✓ Other layer implementations
│   └── models/
│       ├── affinity_predictor.py     # ✓ Simplified inference model
│       └── ...
└── main_simplified.py                # ✓ CLI interface
```

## References

- **Boltz-2 Paper**: [Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
- **Boltz-1 Paper**: [Boltz-1: Open Source Protein Structure and Interaction Prediction](https://doi.org/10.1101/2024.11.19.624167)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}
```

## License

MIT License - Free for academic and commercial use.

## Contributing

Contributions welcome! Focus areas:
- Improved feature engineering for novel protein types
- Ligand handling for larger molecules
- Better MSA generation methods
- API improvements

## Questions?

Open an issue on GitHub or join the community Slack channel.
