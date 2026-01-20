# Instructions to Fix Affinity Prediction on Remote Server

## Problem
The remote server's installed `main_simplified.py` is creating a model with default embedder dimensions (atom_s=128, atom_z=128) but the checkpoint was trained with different dimensions (atom_s=128, atom_feature_dim=388, atom_z=16).

## Solution
Copy the fixed `main_simplified.py` to the remote server:

```bash
# From your local machine:
scp /Users/limcaoco/Projects/dock_boltz/src/boltz/main_simplified.py \
    limcaoco@[remote-host]:/home/limcaoco/opt/anaconda3/envs/boltz_env/lib/python3.12/site-packages/boltz/main_simplified.py
```

Replace `[remote-host]` with your actual remote server hostname.

## What Changed in the Fixed File

The key fix adds a function to infer the correct embedder dimensions from the checkpoint:

1. **New Function: `infer_embedder_args_from_state_dict()`**
   - Extracts actual weights dimensions from checkpoint state_dict
   - Infers: atom_s, atom_feature_dim, atom_z, atom_encoder_heads, atom_encoder_depth
   - These are the ground truth values from the trained model

2. **Updated Prediction Logic**
   - Always infers dimensions from state_dict before creating model
   - Overrides default hparams with inferred values
   - Validates all required parameters were found

3. **Better Error Handling**
   - Validates checkpoint structure
   - Logs warnings about missing/unexpected weight keys
   - Wraps batch processing in try-catch for robustness

## After Copying

Then run your prediction:
```bash
python -m boltz.main_simplified predict \
    --input AA2AR_yaml_configs \
    --output AA2AR_affinity_results/
```

## Why This Works

The checkpoint weights tell us the exact dimensions used during training:
- `embed_atom_features.weight` shape [128, 388] → atom_feature_dim=388
- `embed_atompair_ref_pos.weight` shape [16, 3] → atom_z=16  
- `atom_enc_proj_z.1.weight` shape [12, 16] → atom_encoder_heads=4 (12/(3 depth))

By inferring from the actual weights, the model is created with the correct architecture that matches the trained checkpoint.
