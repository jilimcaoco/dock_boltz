# Using Boltz-2 Affinity Module with External Structures

## Overview

If you have existing 3D structures and sequences from another model, you can:
1. Use the **Boltz-2 evoformer** (MSA + Pairformer) to compute token-level pairwise embeddings
2. Feed those embeddings + your predicted structures to the **affinity module** for binding prediction

This skips the expensive diffusion-based structure prediction while leveraging the learned affinity predictor.

## Pipeline Components You Need

### 1. **Feature Preparation** (`src/boltz/data/feature/featurizerv2.py`)

You must prepare features that match Boltz-2's expected format:

**Minimal Required Features:**

```python
feats = {
    # === Sequence/Token Information ===
    "res_type": torch.tensor([...], dtype=torch.long),           # (B, N_tokens)
    "token_pad_mask": torch.tensor([...], dtype=torch.float),    # (B, N_tokens)
    "mol_type": torch.tensor([...], dtype=torch.long),           # (B, N_tokens) - 0=protein, 2=ligand
    
    # === Atom Information ===
    "atom_pad_mask": torch.tensor([...], dtype=torch.float),     # (B, N_atoms)
    "atom_res_idx": torch.tensor([...], dtype=torch.long),       # (B, N_atoms)
    
    # === MSA Features (for input embedder) ===
    "profile": torch.tensor([...], dtype=torch.float),           # (B, N_tokens, 21)
    "deletion_mean": torch.tensor([...], dtype=torch.float),     # (B, N_tokens, 1)
    
    # === Structure Features ===
    "token_bonds": torch.tensor([...], dtype=torch.float),       # (B, N_tokens, N_tokens, 1)
    
    # === Relative Position Encoding ===
    "relative_position": torch.tensor([...], dtype=torch.long),  # (B, N_tokens, N_tokens)
    "contact_conditioning": torch.tensor([...], dtype=torch.float),  # (B, N_tokens, N_tokens, C)
    "contact_threshold": torch.tensor([...], dtype=torch.float),     # (B, 1, 1)
    
    # === For Atom Encoder ===
    "atom_char": torch.tensor([...], dtype=torch.long),          # (B, N_atoms)
    "atom_pos": torch.tensor([...], dtype=torch.float),          # (B, N_atoms, 3) - optional, may be computed
    
    # === For Affinity Module ===
    "affinity_token_mask": torch.tensor([...], dtype=torch.long), # (B, N_tokens) - marks ligand tokens
    "affinity_mw": torch.tensor([...], dtype=torch.float),       # (B,) - ligand molecular weight
    "profile_affinity": torch.tensor([...], dtype=torch.float),   # (B, N_tokens, 21) - optional
    "deletion_mean_affinity": torch.tensor([...], dtype=torch.float), # (B, N_tokens) - optional
}
```

### 2. **Token Representation Mapping** (for distance computation)

The affinity module uses `token_to_rep_atom` to map tokens to representative atoms:

```python
feats["token_to_rep_atom"] = torch.tensor([...], dtype=torch.float)  
# Shape: (B, N_tokens, N_atoms) - one-hot or soft assignment
# Maps each token to its representative atom(s) for distance calculations
```

### 3. **Input Embedder** (`src/boltz/model/modules/trunkv2.py`)

Converts sequence/token features to sequence embeddings:

```python
from boltz.model.modules.trunkv2 import InputEmbedder

input_embedder = InputEmbedder(
    atom_s=128,                    # Atom embedding dimension
    atom_z=128,                    # Atom pairwise dimension
    token_s=384,                   # Token embedding dimension
    token_z=128,                   # Token pairwise dimension
    atoms_per_window_queries=32,
    atoms_per_window_keys=128,
    atom_feature_dim=128,
    atom_encoder_depth=3,
    atom_encoder_heads=4,
    activation_checkpointing=False,
)

s_inputs = input_embedder(feats)  # (B, N_tokens, token_s)
```

### 4. **Pairformer Module** (MSA + Pairformer Stack)

Computes token-level pairwise embeddings from sequence:

```python
from boltz.model.layers.pairformer import PairformerModule

pairformer = PairformerModule(
    token_s=384,
    token_z=128,
    num_blocks=48,
    num_heads=8,
    # ... other args
)

# Initialize embeddings
s_init = nn.Linear(token_s, token_s)(s_inputs)
z_init = (
    nn.Linear(token_s, token_z)(s_inputs)[:, :, None]
    + nn.Linear(token_s, token_z)(s_inputs)[:, None, :]
)
z_init += relative_position_encoding(feats)
z_init += contact_conditioning(feats)

# Run pairformer
s, z = pairformer(
    s=s_init,
    z=z_init,
    mask=feats["token_pad_mask"].float(),
    pair_mask=pair_mask,
)
# z shape: (B, N_tokens, N_tokens, token_z)
```

### 5. **Affinity Module** (`src/boltz/model/modules/affinity.py`)

Takes embeddings + structures to predict affinity:

```python
from boltz.model.modules.affinity import AffinityModule

affinity_module = AffinityModule(
    token_s=384,
    token_z=128,
    pairformer_args={...},      # Configuration for internal pairformer
    transformer_args={...},      # Configuration for affinity heads
    num_dist_bins=64,
    max_dist=22,
)

# Prepare cross-pair mask (protein-ligand interactions)
pad_token_mask = feats["token_pad_mask"]
rec_mask = (feats["mol_type"] == 0) * pad_token_mask      # Protein tokens
lig_mask = feats["affinity_token_mask"] * pad_token_mask  # Ligand tokens

cross_pair_mask = (
    lig_mask[:, :, None] * rec_mask[:, None, :]
    + rec_mask[:, :, None] * lig_mask[:, None, :]
    + lig_mask[:, :, None] * lig_mask[:, None, :]
)

# Mask the pairwise embeddings to focus on binding interface
z_affinity = z * cross_pair_mask[:, :, :, None]

# Your 3D structure (shape: B, N_atoms, 3)
coords = torch.tensor([...], dtype=torch.float32)

# Run affinity prediction
affinity_out = affinity_module(
    s_inputs=s_inputs.detach(),
    z=z_affinity.detach(),
    x_pred=coords[None, None],  # (1, 1, N_atoms, 3) format for single structure
    feats=feats,
    multiplicity=1,
    use_kernels=False,
)

# Results
affinity_pred_value = affinity_out["affinity_pred_value"]           # log10(IC50)
affinity_logits_binary = affinity_out["affinity_logits_binary"]     # Binder logits
affinity_probability = torch.sigmoid(affinity_logits_binary)        # Binder probability
```

## High-Level Script Outline

```python
import torch
from boltz.model.modules.trunkv2 import InputEmbedder
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.affinity import AffinityModule
from boltz.data.feature.featurizerv2 import FeaturizerV2

# === 1. Prepare Features ===
# Input: sequence, 3D structure, ligand chain ID
featurizer = FeaturizerV2(...)
feats = featurizer(sequence, structure, compute_affinity=True)

# === 2. Compute Evoformer Embeddings ===
input_embedder = InputEmbedder(...)
s_inputs = input_embedder(feats)

# Initialize z
z_init = compute_initial_pairwise_embeddings(s_inputs, feats)

# Run through pairformer for token-level embeddings
pairformer = PairformerModule(...)
s, z = pairformer(s_init, z_init, mask=feats["token_pad_mask"], pair_mask=pair_mask)

# === 3. Run Affinity Module ===
affinity_module = AffinityModule(...)

# Prepare masking
cross_pair_mask = create_cross_pair_mask(feats)
z_affinity = z * cross_pair_mask[:, :, :, None]

# Predict affinity
affinity_out = affinity_module(
    s_inputs=s_inputs.detach(),
    z=z_affinity.detach(),
    x_pred=your_predicted_coords,
    feats=feats,
    multiplicity=1,
)

print(f"Affinity: {affinity_out['affinity_pred_value'].item():.2f} log10(IC50)")
print(f"Binder probability: {torch.sigmoid(affinity_out['affinity_logits_binary']).item():.2%}")
```

## Key Points

### Data Format Requirements

1. **Coordinates**: (B, 1, N_atoms, 3) or (N_atoms, 3) in Ångströms
2. **Token sequences**: Boltz-2 uses token-level (not atom-level) representation
3. **MSA profiles**: Required for sequence embedding - can be simple profiles from alignments
4. **Masks**: Binary or float tensors marking valid tokens/atoms

### What You Can Skip

- **Diffusion model**: Don't need to run structure sampling
- **Template processing**: Can be disabled
- **Structure loss computation**: Only needed during training
- **Full recycling**: Can use single forward pass without recycling steps

### What You Must Have

- **Pairformer weights**: Needed for evoformer embeddings
- **Affinity module weights**: Needed for affinity prediction
- **MSA features**: Required by input embedder (can be synthetic/from external alignments)
- **Proper feature formatting**: Must match Boltz-2's expectations

### Computational Cost

- Input embedder: ~10% of total cost
- Pairformer (evoformer): ~80-90% of total cost
- Affinity module: ~10% of total cost

Compared to full pipeline: You save the expensive diffusion sampling (~50% of time) and structure losses.

## Implementation Challenges

1. **Feature Engineering**: Most complex part - mapping external structures to Boltz-2 feature format
2. **MSA Features**: Need good MSA profiles for sequence embedding quality
3. **Token-to-atom mapping**: Must correctly compute `token_to_rep_atom` for distance calculations
4. **Coordinate system**: Ensure coordinates are centered/normalized appropriately

## Related Files to Check

- [src/boltz/data/feature/featurizerv2.py](src/boltz/data/feature/featurizerv2.py) - Feature preparation (complex, ~2400 lines)
- [src/boltz/model/modules/trunkv2.py](src/boltz/model/modules/trunkv2.py) - Input embedder, pairformer initialization
- [src/boltz/model/layers/pairformer.py](src/boltz/model/layers/pairformer.py) - Pairformer implementation
- [src/boltz/model/modules/affinity.py](src/boltz/model/modules/affinity.py) - Affinity module
- [src/boltz/model/models/boltz2.py](src/boltz/model/models/boltz2.py) - Full pipeline reference (lines 410-490)
