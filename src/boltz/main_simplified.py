"""
Simplified CLI for affinity prediction with external structures.

Usage:
    python -m boltz.main predict --input config.yaml --output results.csv --checkpoint model.ckpt
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import click
import torch
import yaml

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.model.models.affinity_predictor import AffinityPredictor

logger = logging.getLogger(__name__)

# Model checkpoint URLs
BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


def get_cache_path() -> Path:
    """Get the cache directory for model checkpoints."""
    cache_dir = Path.home() / ".boltz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def infer_embedder_args_from_state_dict(state_dict: dict) -> dict:
    """Infer embedder_args from checkpoint state_dict dimensions.
    
    Parameters
    ----------
    state_dict : dict
        Model state dict from checkpoint
        
    Returns
    -------
    dict
        Inferred embedder arguments
    """
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
    # Shape should be [atom_encoder_depth * atom_encoder_heads, atom_z]
    if "input_embedder.atom_enc_proj_z.1.weight" in state_dict:
        proj_z_out_dim = state_dict["input_embedder.atom_enc_proj_z.1.weight"].shape[0]
        # Assuming atom_encoder_depth is 3 (count p_mlp layers)
        atom_encoder_depth = 3
        atom_encoder_heads = proj_z_out_dim // atom_encoder_depth
        embedder_args["atom_encoder_depth"] = atom_encoder_depth
        embedder_args["atom_encoder_heads"] = atom_encoder_heads
        logger.debug(f"Inferred atom_encoder_depth={atom_encoder_depth}, atom_encoder_heads={atom_encoder_heads} from checkpoint")
    
    # Set default window sizes if not found (these are architectural constants)
    if "atoms_per_window_queries" not in embedder_args:
        embedder_args["atoms_per_window_queries"] = 32
    
    if "atoms_per_window_keys" not in embedder_args:
        embedder_args["atoms_per_window_keys"] = 128
    
    # Activation checkpointing is typically False for inference
    if "activation_checkpointing" not in embedder_args:
        embedder_args["activation_checkpointing"] = False
    
    logger.debug(f"Full inferred embedder_args: {embedder_args}")
    return embedder_args


def download_checkpoint(
    urls: list[str],
    output_path: Path,
    description: str = "checkpoint",
) -> Path:
    """Download checkpoint from URLs with fallback.
    
    Parameters
    ----------
    urls : list[str]
        List of URLs to try in order
    output_path : Path
        Where to save the checkpoint
    description : str
        Description for logging
        
    Returns
    -------
    Path
        Path to the downloaded checkpoint
    """
    if output_path.exists():
        logger.info(f"Found existing {description} at {output_path}")
        return output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for i, url in enumerate(urls):
        try:
            logger.info(f"Downloading {description} from {url}")
            urllib.request.urlretrieve(url, str(output_path))  # noqa: S310
            logger.info(f"Successfully downloaded {description} to {output_path}")
            return output_path
        except Exception as e:  # noqa: BLE001
            if i == len(urls) - 1:
                msg = f"Failed to download {description} from all URLs. Last error: {e}"
                raise RuntimeError(msg) from e
            logger.warning(f"Failed to download from {url}, trying next URL...")
            continue
    
    return output_path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


@click.group()
def cli():
    """Boltz-2 Affinity Predictor CLI."""
    pass


@cli.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input YAML config or directory of configs",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output CSV file with predictions",
)
@click.option(
    "--checkpoint",
    required=False,
    type=click.Path(exists=False),
    default=None,
    help="Path to model checkpoint. If not provided, will download boltz2_aff.ckpt",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use (cuda/cpu)",
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Batch size for inference",
)
@click.option(
    "--recycling-steps",
    default=0,
    type=int,
    help="Number of recycling steps for pairformer",
)
def predict(
    input: str,
    output: str,
    checkpoint: Optional[str],
    device: str,
    batch_size: int,
    recycling_steps: int,
):
    """Predict binding affinity for protein-ligand complexes.

    Input should be a YAML config or directory containing YAML configs
    describing the protein sequences and 3D structures.

    Example config (affinity.yaml):
    ```
    version: 1
    sequences:
      - protein:
          id: A
          sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQ...
      - ligand:
          id: B
          smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
    properties:
      - affinity:
          binder: B
    ```
    """
    # Download checkpoint if not provided
    if checkpoint is None:
        cache_dir = get_cache_path()
        checkpoint_path = cache_dir / "boltz2_aff.ckpt"
        checkpoint = str(download_checkpoint(
            BOLTZ2_AFFINITY_URL_WITH_FALLBACK,
            checkpoint_path,
            description="Boltz-2 affinity checkpoint",
        ))
    
    logger.info(f"Loading checkpoint from {checkpoint}")
    ckpt = load_checkpoint(checkpoint, device=device)

    # Extract model args from checkpoint
    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", {})
    
    # Validate checkpoint structure
    if not state_dict:
        msg = "Checkpoint does not contain 'state_dict'. Invalid checkpoint file."
        raise ValueError(msg)
    
    # Log checkpoint structure for debugging
    logger.debug(f"Checkpoint keys: {list(ckpt.keys())}")
    logger.debug(f"Hyperparameters keys: {list(hparams.keys())}")

    # Initialize model
    logger.info("Initializing affinity predictor")
    
    # Prepare affinity_model_args with both pairformer and transformer args
    token_s = hparams.get("token_s", 384)
    token_z = hparams.get("token_z", 128)
    
    # Extract embedder_args from checkpoint, or infer from state_dict
    embedder_args = hparams.get("embedder_args", {})
    
    # Always infer from state_dict to ensure we get the correct dimensions
    # This handles cases where hparams has incomplete or default values
    inferred_embedder_args = infer_embedder_args_from_state_dict(state_dict)
    
    # Prefer inferred values over hparams (inferred values are ground truth from weights)
    embedder_args = {**embedder_args, **inferred_embedder_args}
    logger.info(f"Inferred embedder args from checkpoint: {inferred_embedder_args}")
    
    # Validate critical parameters were inferred
    required_params = ["atom_s", "atom_z", "atom_feature_dim", "atom_encoder_heads"]
    missing = [p for p in required_params if p not in embedder_args]
    if missing:
        msg = f"Failed to infer required embedder parameters: {missing}. Checkpoint may be incompatible."
        raise ValueError(msg)
    
    affinity_model_args = hparams.get("affinity_model_args", {})
    
    # Ensure transformer_args exists in affinity_model_args
    if "transformer_args" not in affinity_model_args:
        affinity_model_args["transformer_args"] = {}
    # Ensure transformer_args has token_s
    if "token_s" not in affinity_model_args["transformer_args"]:
        affinity_model_args["transformer_args"]["token_s"] = token_s
    
    logger.debug(f"token_s={token_s}, token_z={token_z}")
    logger.debug(f"embedder_args={embedder_args}")
    
    model = AffinityPredictor(
        token_s=token_s,
        token_z=token_z,
        embedder_args=embedder_args,
        msa_args=hparams.get("msa_args", {}),
        pairformer_args=hparams.get("pairformer_args", {}),
        affinity_model_args=affinity_model_args,
        affinity_ensemble=hparams.get("affinity_ensemble", False),
        affinity_mw_correction=hparams.get("affinity_mw_correction", True),
    )
    # Load state dict with warnings about missing/unexpected keys
    load_result = model.load_state_dict(ckpt["state_dict"], strict=False)
    if load_result.missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {load_result.missing_keys[:5]}...")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {load_result.unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()

    # Initialize data module
    logger.info(f"Loading input from {input}")
    input_path = Path(input)
    data_module = Boltz2InferenceDataModule(
        input_dir=input_path,
        batch_size=batch_size,
        num_workers=0,
    )
    data_module.setup("predict")
    predict_dataloader = data_module.predict_dataloader()

    # Run predictions
    logger.info(f"Running predictions ({len(predict_dataloader)} batches)")
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(predict_dataloader):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get 3D coordinates from batch
                coords = batch.get("coords")
                if coords is None:
                    logger.error(f"Batch {batch_idx} missing 'coords' field. Ensure your input includes 3D structures.")
                    continue

                # Remove singleton dimension if present
                if coords.dim() == 4:
                    coords = coords.squeeze(1)

                # Run model
                out = model(
                    feats=batch,
                    coords=coords,
                    recycling_steps=recycling_steps,
                )

                # Extract results
                batch_results = {
                    "complex_id": batch.get("complex_id", [f"complex_{batch_idx}_0"]),
                    "affinity_pred_value": out["affinity_pred_value"].cpu().numpy(),
                    "affinity_probability_binary": out["affinity_probability_binary"].cpu().numpy(),
                }
                results.append(batch_results)

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(predict_dataloader)} batches")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                logger.debug(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'not a dict'}")
                # Continue with next batch instead of failing completely
                continue

    # Write results
    logger.info(f"Writing results to {output}")
    
    if not results:
        logger.warning("No results to write. All batches may have failed or been skipped.")
        # Still create an empty CSV with headers
        import csv
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["complex_id", "affinity_pred_value", "affinity_probability_binary"])
        return
    
    import csv
    import numpy as np

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["complex_id", "affinity_pred_value", "affinity_probability_binary"])

        for batch_result in results:
            for i, cid in enumerate(batch_result["complex_id"]):
                affinity_value = batch_result["affinity_pred_value"][i, 0]
                affinity_prob = batch_result["affinity_probability_binary"][i, 0]
                writer.writerow([cid, f"{affinity_value:.4f}", f"{affinity_prob:.4f}"])

    logger.info(f"Results saved to {output}. Total predictions: {sum(len(r['complex_id']) for r in results)}")


@cli.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Input YAML config",
)
def validate(input: str):
    """Validate input configuration file."""
    try:
        config = load_config(input)
        click.echo(f"✓ Config is valid")
        click.echo(f"  Sequences: {len(config.get('sequences', []))}")
        click.echo(f"  Properties: {len(config.get('properties', []))}")
    except Exception as e:
        click.echo(f"✗ Config is invalid: {e}", err=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cli()
