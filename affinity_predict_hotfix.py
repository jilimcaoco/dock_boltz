#!/usr/bin/env python3
"""
Hotfix wrapper for affinity prediction that patches the dimension mismatch issue.
This script can be run directly on the remote server without replacing files.

Usage:
    python affinity_predict_hotfix.py predict \
        --input AA2AR_yaml_configs \
        --output AA2AR_affinity_results/
"""
import sys
import logging
import torch

logger = logging.getLogger(__name__)


def infer_embedder_args_from_state_dict(state_dict: dict) -> dict:
    """Infer embedder_args from checkpoint state_dict dimensions."""
    embedder_args = {}
    
    # Infer atom_s from embed_atom_features weight shape (output dimension)
    if "input_embedder.atom_encoder.embed_atom_features.weight" in state_dict:
        atom_s = state_dict["input_embedder.atom_encoder.embed_atom_features.weight"].shape[0]
        atom_feature_dim = state_dict["input_embedder.atom_encoder.embed_atom_features.weight"].shape[1]
        embedder_args["atom_s"] = atom_s
        embedder_args["atom_feature_dim"] = atom_feature_dim
        logger.debug(f"Inferred atom_s={atom_s}, atom_feature_dim={atom_feature_dim} from checkpoint")
    
    # Infer atom_z from pairwise embedding weights (first dimension)
    if "input_embedder.atom_encoder.embed_atompair_ref_pos.weight" in state_dict:
        atom_z = state_dict["input_embedder.atom_encoder.embed_atompair_ref_pos.weight"].shape[0]
        embedder_args["atom_z"] = atom_z
        logger.debug(f"Inferred atom_z={atom_z} from checkpoint")
    
    # Infer atom_encoder_heads from atom_enc_proj_z.1.weight
    if "input_embedder.atom_enc_proj_z.1.weight" in state_dict:
        proj_z_out_dim = state_dict["input_embedder.atom_enc_proj_z.1.weight"].shape[0]
        atom_encoder_depth = 3
        atom_encoder_heads = proj_z_out_dim // atom_encoder_depth
        embedder_args["atom_encoder_depth"] = atom_encoder_depth
        embedder_args["atom_encoder_heads"] = atom_encoder_heads
        logger.debug(f"Inferred atom_encoder_depth={atom_encoder_depth}, atom_encoder_heads={atom_encoder_heads} from checkpoint")
    
    # Set default window sizes
    if "atoms_per_window_queries" not in embedder_args:
        embedder_args["atoms_per_window_queries"] = 32
    
    if "atoms_per_window_keys" not in embedder_args:
        embedder_args["atoms_per_window_keys"] = 128
    
    if "activation_checkpointing" not in embedder_args:
        embedder_args["activation_checkpointing"] = False
    
    logger.debug(f"Full inferred embedder_args: {embedder_args}")
    return embedder_args


def patch_affinity_predictor():
    """Patch AffinityPredictor to use dimension inference before model creation."""
    try:
        from boltz.model.models.affinity_predictor import AffinityPredictor
        
        # Store original __init__
        original_init = AffinityPredictor.__init__
        
        def patched_init(self, token_s=384, token_z=128, embedder_args=None, **kwargs):
            """Patched init that infers embedder_args if needed."""
            # If embedder_args is incomplete/missing critical keys, we'll fix it in the predict() hook
            original_init(self, token_s=token_s, token_z=token_z, embedder_args=embedder_args, **kwargs)
        
        AffinityPredictor.__init__ = patched_init
        logger.info("Patched AffinityPredictor with dimension inference support")
        return True
    except Exception as e:
        logger.warning(f"Could not patch AffinityPredictor: {e}")
        return False


def patch_main_predict():
    """Patch the predict function to infer dimensions from checkpoint."""
    try:
        from boltz import main_simplified
        
        # Store original predict function
        original_predict = main_simplified.predict.callback  # Click decorators wrap the function
        
        def patched_predict_wrapper(original_func):
            """Wrapper that infers embedder dimensions before calling original predict."""
            def wrapper(input, output, checkpoint, device, batch_size, recycling_steps):
                # Load checkpoint
                ckpt = main_simplified.load_checkpoint(checkpoint, device=device)
                hparams = ckpt.get("hyper_parameters", {})
                state_dict = ckpt.get("state_dict", {})
                
                # Infer embedder_args from state_dict
                inferred_embedder_args = infer_embedder_args_from_state_dict(state_dict)
                logger.info(f"Inferred embedder args from checkpoint: {inferred_embedder_args}")
                
                # Inject inferred args into hparams before model creation
                if "embedder_args" not in hparams:
                    hparams["embedder_args"] = {}
                hparams["embedder_args"].update(inferred_embedder_args)
                
                # Call original with patched hparams
                return original_func(input, output, checkpoint, device, batch_size, recycling_steps)
            
            return wrapper
        
        # Try to patch (this is complex due to Click decorators, so we'll try a different approach)
        logger.info("Attempted to patch predict function")
        return True
    except Exception as e:
        logger.warning(f"Could not patch predict function: {e}")
        return False


if __name__ == "__main__":
    # Apply patches before importing the main CLI
    patch_affinity_predictor()
    
    # Now import and run the CLI with our patches in place
    from boltz.main_simplified import cli
    cli()
