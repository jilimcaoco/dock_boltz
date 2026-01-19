# Boltz-2 Affinity Prediction Module Guide

## Overview

The affinity prediction module in Boltz-2 is designed to predict protein-ligand binding affinity from predicted 3D structures and evoformer embeddings. It produces two main outputs:

1. **`affinity_pred_value`**: A log10(IC50) value predicting the binding affinity in μM (suitable for ligand optimization)
2. **`affinity_probability_binary`**: A probability score (0-1) indicating whether a molecule is a binder (suitable for hit discovery/screening)

## Architecture Overview

The affinity module operates on the output of the main structure prediction pipeline, specifically using:
- **Pairwise embeddings (z)** from the evoformer layer
- **Predicted 3D coordinates** from the diffusion model
- **Sequence embeddings (s_inputs)** preprocessed for affinity
- **Token masks** distinguishing receptor, ligand, and padding tokens

## Key Components

### 1. AffinityModule (`src/boltz/model/modules/affinity.py`)

The core `AffinityModule` is composed of several key components:

#### Input Conditioning
```python
# Gaussian distance smearing - converts pairwise distances to features
self.dist_bin_pairwise_embed: Embedding(num_dist_bins, token_z)

# Sequence to pairwise embedding projections
self.s_to_z_prod_in1: LinearNoBias(token_s, token_z)  # First dimension
self.s_to_z_prod_in2: LinearNoBias(token_s, token_z)  # Second dimension
```

#### Processing Pipeline
1. **Distance Discretization**: Pairwise distances between representative atoms are binned into 64 distance bins (default, range 2-22 Å)
2. **Embedding Conditioning**: 
   - Normalizes and projects the evoformer pairwise embeddings (`z`)
   - Adds sequence information via outer product: `z + s_to_z_prod_in1(s)[:,:,None,:] + s_to_z_prod_in2(s)[:,None,:,:]`
   - Adds distance information via PairwiseConditioning

3. **Pairformer Stack**: Processes the conditioned embeddings with `PairformerNoSeqModule`

4. **Affinity Heads**: Extracts predictions from the processed embeddings

#### Technical Details

**Distance Binning:**
```
Inputs:  Pairwise distances d computed from representative atom coordinates
         Boundaries at: torch.linspace(2, 22, 63)  # 64 bins total
Output:  One-hot encoded distance bins, embedded to token_z dimension
```

**Cross-Pair Masking:**
The module uses a cross-pair mask to focus on protein-ligand interactions:
```
cross_pair_mask = (lig_mask[:, None] * rec_mask[None, :]   # lig-rec pairs
                 + rec_mask[:, None] * lig_mask[None, :]   # rec-lig pairs  
                 + lig_mask[:, None] * lig_mask[None, :])  # lig-lig pairs
```

This mask zeros out protein-protein interactions, focusing the module on binding-relevant information.

### 2. AffinityHeadsTransformer

Converts processed embeddings to final predictions:

```python
# Aggregation: Weighted average of cross-pair embeddings
g = sum(z * cross_pair_mask) / sum(cross_pair_mask)

# MLP projection to create a global representation
g = affinity_out_mlp(g)  # token_z -> token_s

# Two parallel prediction heads
affinity_pred_value = to_affinity_pred_value(g)          # Regression output
affinity_logits_binary = to_affinity_logits_binary(
    to_affinity_pred_score(g)
)  # Binary classification logits
```

**Key insight**: The module averages pairwise embeddings over all protein-ligand interaction pairs to create a single global representation for the complex.

## Data Flow in the Main Model

### Cropping (AffinityCropper)

Before affinity prediction, proteins are spatially cropped around the ligand:

- **Location**: `src/boltz/data/crop/affinity.py`
- **Strategy**: Selects tokens based on minimum distance to ligand (spatial cropping)
- **Parameters**:
  - `neighborhood_size`: Controls balance between spatial and sequential cropping
  - `max_tokens_protein`: Maximum protein tokens to include (default 200)
  - `max_atoms`: Maximum total atoms to consider

This ensures the model only processes locally relevant context.

### Feature Preparation

The featurizer (`src/boltz/data/feature/featurizerv2.py`) prepares affinity-specific features:

1. **Token Mask**: `affinity_token_mask` marks which tokens belong to the ligand
2. **Molecular Weight**: `affinity_mw` - the ligand's molecular weight (used for post-hoc correction)
3. **MSA Features**: Special MSA features are computed if affinity computation is enabled
4. **Mol Type**: `mol_type == const.chain_type_ids["NONPOLYMER"]` identifies ligand tokens

### Inference Pipeline

In `src/boltz/model/models/boltz2.py`, the affinity prediction occurs after structure prediction:

```python
# 1. Extract best structure from diffusion sampling (by pIPTM score)
argsort = torch.argsort(dict_out["iptm"], descending=True)
best_idx = argsort[0].item()
coords_affinity = dict_out["sample_atom_coords"][best_idx][None, None]

# 2. Create affinity-specific embeddings
s_inputs = self.input_embedder(feats, affinity=True)

# 3. Prepare pairwise mask (receptor-ligand interactions)
cross_pair_mask = (lig_mask[:, None] * rec_mask[None, :]
                 + rec_mask[:, None] * lig_mask[None, :]
                 + lig_mask[:, None] * lig_mask[None, :])
z_affinity = z * cross_pair_mask[None, :, :, None]

# 4. Run affinity module
dict_out_affinity = self.affinity_module(
    s_inputs=s_inputs.detach(),
    z=z_affinity.detach(),
    x_pred=coords_affinity,
    feats=feats,
    multiplicity=1,
    use_kernels=self.use_kernels,
)

# 5. Apply sigmoid to convert logits to probability
affinity_probability_binary = torch.sigmoid(
    dict_out_affinity["affinity_logits_binary"]
)
```

## Outputs

The module produces:

1. **`affinity_pred_value`** (shape: [batch, 1])
   - Log10(IC50) scale
   - Predicted binding affinity in μM
   - Use case: Hit-to-lead, ligand optimization

2. **`affinity_logits_binary`** (shape: [batch, 1])
   - Raw logits for binary binder/non-binder classification
   - Applied sigmoid → `affinity_probability_binary`
   - Use case: Hit discovery, screening

3. **`affinity_probability_binary`** (shape: [batch, 1])
   - Sigmoid-activated probability (0-1)
   - Predicted probability that molecule is a binder
   - Use case: Ranking and filtering

## Ensemble Mode

When `affinity_ensemble=True`, two separate `AffinityModule` instances are run:

```python
# Average predictions from both models
affinity_pred_value = (pred1 + pred2) / 2
affinity_probability_binary = (prob1 + prob2) / 2

# Apply molecular weight correction (empirically derived coefficients)
if affinity_mw_correction:
    model_coef = 1.03525938
    mw_coef = -0.59992683
    bias = 2.83288489
    mw = feats["affinity_mw"][0] ** 0.3
    affinity_pred_value = (model_coef * affinity_pred_value 
                          + mw_coef * mw 
                          + bias)
```

The molecular weight correction is an empirical post-hoc adjustment that accounts for ligand size biases in the training data.

## Input Requirements

### Ligand Constraints
- Must be a single non-polymer chain (ligand)
- Heavy atom count (excluding hydrogens): **max 128 atoms**
- Recommended max: **56 atoms** (limit during training)
- Must include all necessary atoms for binding representation

### Configuration (YAML)
```yaml
properties:
  - affinity:
      binder: <CHAIN_ID>  # Chain ID of the ligand
```

## Key Architectural Decisions

1. **Distance-Based Conditioning**: Rather than using only sequence/embedding-based features, the module explicitly conditions on computed 3D distances from predicted coordinates

2. **Cross-Pair Focus**: Masking keeps the model focused on protein-ligand interactions, reducing computational burden and improving signal

3. **Post-Structure Integration**: Affinity prediction uses the best-predicted structure (selected by pIPTM) rather than a diffusion ensemble, simplifying computation

4. **Dual Outputs**: Two separate training objectives (regression for IC50 values, classification for binder probability) allow flexibility in downstream applications

5. **Molecular Weight Correction**: Empirical correction applied to ensemble predictions to account for ligand size effects observed in the training data

## Representative Atom Computation

The module uses `token_to_rep_atom` features to map tokens to representative atoms:

```python
x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
```

This allows computing token-level pairwise distances from atom coordinates, effectively creating a coarse-grained distance matrix at the token level.

## Performance Notes

- The affinity module is relatively lightweight compared to the main structure prediction pipeline
- Runs only once per prediction (on best structure) rather than over diffusion ensemble
- Distance computation uses efficient `torch.cdist`
- Can be torch.compiled for optimization on compatible hardware

## Related Files

- **Core Module**: [src/boltz/model/modules/affinity.py](src/boltz/model/modules/affinity.py)
- **Data Cropping**: [src/boltz/data/crop/affinity.py](src/boltz/data/crop/affinity.py)
- **Feature Preparation**: [src/boltz/data/feature/featurizerv2.py](src/boltz/data/feature/featurizerv2.py)
- **Main Model Integration**: [src/boltz/model/models/boltz2.py](src/boltz/model/models/boltz2.py) (lines 608-720)
- **Example Config**: [examples/affinity.yaml](examples/affinity.yaml)
