# Cleanup Plan for Boltz-2 Affinity Prediction Workflow

## REMOVE (No Longer Needed)

### Training & Evaluation Scripts
- `/scripts/train/` - Training pipeline (no longer needed)
- `/scripts/eval/` - Evaluation scripts (no longer needed)
- `/scripts/process/` - Data processing scripts (optional, keep if useful for feature prep)

### Model Modules - Remove These
- `src/boltz/model/modules/diffusion.py` - Old diffusion implementation
- `src/boltz/model/modules/diffusionv2.py` - Structure prediction
- `src/boltz/model/modules/diffusion_conditioning.py` - Conditioning for diffusion
- `src/boltz/model/modules/confidence.py` - Old confidence module
- `src/boltz/model/modules/confidence_utils.py` - Confidence utilities
- `src/boltz/model/modules/confidencev2.py` - Confidence prediction
- `src/boltz/model/modules/transformers.py` - Old transformer implementation
- `src/boltz/model/modules/trunk.py` - Old trunk implementation
- `src/boltz/model/modules/encoders.py` - Old encoder implementation

### Loss Functions - Remove All
- `src/boltz/model/loss/` - All training loss functions

### Optimization - Remove
- `src/boltz/model/optim/` - Optimizers and schedulers for training

### Potentials - Remove
- `src/boltz/model/potentials/` - Physical potentials (only used in diffusion steering)

### Full Models - Remove
- `src/boltz/model/models/boltz1.py` - Old Boltz-1 model
- `src/boltz/model/models/boltz2.py` - Full Boltz-2 (replace with simple inference wrapper)

### Documentation
- `docs/training.md` - Training guide (no longer needed)
- `docs/evaluation.md` - Evaluation guide (no longer needed)

### Examples
- `examples/cyclic_prot.yaml` - Structure prediction config
- `examples/multimer.yaml` - Structure prediction config
- `examples/prot_custom_msa.yaml` - Structure prediction config
- `examples/prot_no_msa.yaml` - Structure prediction config
- `examples/prot.yaml` - Structure prediction config
- `examples/pocket.yaml` - Structure prediction config
- `examples/ligand.fasta` - Ligand sequence (keep affinity.yaml only)
- `examples/prot.fasta` - Protein sequence (keep ligand.fasta in affinity context)
- `examples/msa/` - MSA examples (optional to keep for reference)
- `examples/README.md` - Update or rewrite for affinity workflow

### Tests
- `tests/test_regression.py` - Regression tests (relies on full model)
- Remove test cases for diffusion, confidence, training, etc.

---

## KEEP & SIMPLIFY

### Data Module
- `src/boltz/data/` - Core data structures
  - Keep: `const.py`, `types.py`, `mol.py`, `pad.py`
  - Keep: `feature/featurizerv2.py` (with removal of training-specific features)
  - Keep: `crop/affinity.py`
  - Keep: `parse/` (for parsing input structures)
  - Keep: `tokenize/boltz2.py`
  - Remove: `msa/mmseqs2.py` (external dependency, keep for reference)
  - Remove: `write/` (writing predictions, maybe simplify)

### Model Layers
- `src/boltz/model/layers/` - Core neural network layers
  - Keep all basic layers
  - Remove: `dropout.py` if only used in training

### Model Modules - Keep These
- `src/boltz/model/modules/affinity.py` - **CORE: Affinity prediction**
- `src/boltz/model/modules/trunkv2.py` - **CORE: Input embedder, pairwise initialization**
- `src/boltz/model/modules/encodersv2.py` - **CORE: Attention encoders**
- `src/boltz/model/modules/transformersv2.py` - **CORE: Token transformer**
- `src/boltz/model/modules/utils.py` - Utility functions

### Core Architecture
- `src/boltz/model/layers/pairformer.py` - **CORE: Evoformer/Pairformer**
- `src/boltz/model/layers/` - All other layer implementations

### Main Entry Point
- `src/boltz/main.py` - Simplify to only support affinity prediction workflow

### Config & Examples
- `pyproject.toml` - Keep but update dependencies
- `README.md` - Update with new purpose
- `examples/affinity.yaml` - Keep as primary example
- `docs/prediction.md` - Keep and update for affinity-only workflow

---

## Summary of Changes

| Category | Action |
|----------|--------|
| Diffusion code | DELETE |
| Confidence code | DELETE |
| Training code | DELETE |
| Evaluation code | DELETE |
| Loss functions | DELETE |
| Optimizers | DELETE |
| Physical potentials | DELETE |
| Old models | DELETE |
| Old docs | DELETE |
| Data/Feature code | KEEP |
| Layers | KEEP |
| Affinity module | KEEP |
| Evoformer (Pairformer) | KEEP |
| Input embedder | KEEP |
| Main entry point | SIMPLIFY |
| Examples | SIMPLIFY |

This should reduce the codebase by ~50-60% and make it much clearer what components are actually being used.
