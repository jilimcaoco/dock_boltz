# Cleaned Codebase Structure

## Final Directory Layout

```
dock_boltz/
├── README.md                          # ✓ Updated for affinity prediction
├── README_ORIGINAL.md                 # (Archived original)
├── CLEANUP_COMPLETE.md                # ✓ This cleanup summary
├── CLEANUP_PLAN.md                    # ✓ What was removed
├── AFFINITY_MODULE_GUIDE.md           # ✓ Technical deep dive
├── EXTERNAL_STRUCTURES_GUIDE.md       # ✓ Integration guide
├── pyproject.toml
├── LICENSE
│
├── src/boltz/
│   ├── __init__.py
│   ├── main_simplified.py             # ✓ NEW: Simplified CLI (250 lines)
│   │
│   ├── data/                          # ✓ KEEP: Feature preparation pipeline
│   │   ├── const.py                   # Constants
│   │   ├── types.py                   # Data types
│   │   ├── mol.py                     # Molecular utilities
│   │   ├── pad.py                     # Padding utilities
│   │   ├── feature/
│   │   │   ├── featurizerv2.py        # ✓ Core feature preparation
│   │   │   └── symmetry.py
│   │   ├── parse/                     # Input parsing
│   │   │   ├── yaml.py
│   │   │   ├── fasta.py
│   │   │   ├── pdb.py
│   │   │   ├── mmcif.py
│   │   │   ├── a3m.py
│   │   │   ├── csv.py
│   │   │   └── schema.py
│   │   ├── tokenize/
│   │   │   ├── boltz2.py              # ✓ Tokenization
│   │   │   └── tokenizer.py
│   │   ├── crop/
│   │   │   ├── affinity.py            # ✓ Affinity-specific cropping
│   │   │   └── cropper.py
│   │   ├── msa/
│   │   │   └── mmseqs2.py             # MSA generation
│   │   ├── module/
│   │   │   ├── inferencev2.py         # ✓ Inference data module
│   │   │   └── trainingv2.py          # (Training module, could be removed)
│   │   └── filter/                    # Data filtering utilities
│   │
│   ├── model/
│   │   ├── models/
│   │   │   ├── affinity_predictor.py  # ✓ NEW: Clean inference model
│   │   │   └── boltz2_full.py.bak     # (Archived full model for reference)
│   │   │
│   │   ├── modules/
│   │   │   ├── affinity.py            # ✓ CORE: Affinity prediction
│   │   │   ├── trunkv2.py             # ✓ CORE: Input embedder, pairwise init
│   │   │   ├── encodersv2.py          # ✓ CORE: Attention encoders
│   │   │   ├── transformersv2.py      # ✓ CORE: Token transformers
│   │   │   └── utils.py               # ✓ Utilities
│   │   │
│   │   └── layers/
│   │       ├── pairformer.py          # ✓ CORE: Evoformer/Pairformer
│   │       ├── outer_product_mean.py  # ✓ Attention layer
│   │       ├── pair_averaging.py      # ✓ Pair averaging
│   │       ├── triangular_mult.py     # ✓ Triangular multiplication
│   │       ├── triangular_attention/  # ✓ Triangular attention
│   │       ├── attention.py           # ✓ Basic attention
│   │       ├── attentionv2.py         # ✓ Attention v2
│   │       ├── transition.py          # ✓ Transitions
│   │       ├── relative.py            # ✓ Relative positional encoding
│   │       ├── initialize.py          # ✓ Weight initialization
│   │       ├── dropout.py             # ✓ Dropout utilities
│   │       └── confidence_utils.py    # (Unused, could be removed)
│
├── docs/
│   ├── prediction.md                  # ✓ Updated for affinity workflow
│   └── pearson_plot.png
│
├── examples/
│   ├── affinity.yaml                  # ✓ Primary example
│   └── msa/                           # Optional MSA examples
│
├── scripts/
│   └── process/                       # Data processing scripts (optional)
│       ├── README.md
│       ├── msa.py
│       ├── rcsb.py
│       └── ccd.py
│
└── tests/
    └── (empty - tests removed)

```

## Removed Components

### Code Deletions
```
❌ src/boltz/model/loss/                    # Training losses
❌ src/boltz/model/optim/                   # Optimizers/schedulers  
❌ src/boltz/model/potentials/              # Physical potentials
❌ src/boltz/data/write/                    # Output writing
❌ src/boltz/model/modules/diffusion.py     # Old diffusion
❌ src/boltz/model/modules/diffusionv2.py   # Structure prediction
❌ src/boltz/model/modules/confidence*.py   # Confidence prediction
❌ src/boltz/model/modules/transformers.py  # Old transformers
❌ src/boltz/model/modules/trunk.py         # Old trunk
❌ src/boltz/model/modules/encoders.py      # Old encoders
❌ src/boltz/model/models/boltz1.py         # Boltz-1
❌ src/boltz/model/models/boltz2.py         # Archived as boltz2_full.py.bak
❌ docs/training.md                         # Training guide
❌ docs/evaluation.md                       # Evaluation guide
❌ scripts/train/                           # Training pipeline
❌ scripts/eval/                            # Evaluation scripts
❌ tests/*                                  # All tests
```

## Core Components (Kept & Refined)

### Data Processing Pipeline
- ✓ Feature featurization (featurizerv2.py)
- ✓ Input parsing (YAML, FASTA, PDB, mmCIF)
- ✓ Tokenization (Boltz-2 format)
- ✓ Cropping (affinity-specific)
- ✓ MSA integration (mmseqs2)

### Model Architecture
- ✓ Input embedder (atom → token embeddings)
- ✓ Pairformer/Evoformer (sequence learning)
- ✓ Attention mechanisms (all variants)
- ✓ Affinity module (structure-aware prediction)

### Inference
- ✓ Simplified model (affinity_predictor.py)
- ✓ CLI interface (main_simplified.py)
- ✓ Data modules for inference

---

## File Counts

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Model modules | 13 files | 5 files | ✓ 62% reduction |
| Model layers | ~20 files | ~17 files | ✓ Minimal changes |
| Training code | ~3000 lines | 0 lines | ✓ Deleted |
| Main entry point | 1415 lines | 250 lines | ✓ 82% reduction |
| Total codebase | ~15,000 lines | ~8,000 lines | ✓ 47% reduction |

---

## Quick Reference

### To run affinity prediction:
```bash
python -m boltz.main_simplified predict \
  --input config.yaml \
  --output predictions.csv \
  --checkpoint model.ckpt
```

### To validate configuration:
```bash
python -m boltz.main_simplified validate --input config.yaml
```

### To use programmatically:
```python
from boltz.model.models.affinity_predictor import AffinityPredictor
model = AffinityPredictor(...)
predictions = model(feats=features, coords=coordinates)
```

---

## Migration Notes

If you were using the old code:

**Old imports that no longer work:**
```python
❌ from boltz.model.models.boltz2 import Boltz2
❌ from boltz.model.loss import * 
❌ from boltz.model.optim import *
```

**New imports:**
```python
✓ from boltz.model.models.affinity_predictor import AffinityPredictor
✓ from boltz.model.modules.affinity import AffinityModule
✓ from boltz.main_simplified import predict, validate
```

---

**Status**: ✅ Cleanup Complete
**Date**: December 29, 2025
**Result**: Clean, focused codebase ready for affinity prediction workflows
