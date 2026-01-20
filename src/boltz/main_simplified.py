"""
Simplified CLI for affinity prediction with full feature preparation.

This module provides a complete pipeline for affinity prediction that includes:
- Checkpoint loading with automatic dimension inference
- YAML configuration parsing
- Feature preparation from sequences and 3D structures
- Model inference

Usage:
    python -m boltz.main_simplified predict --input config.yaml --output results.csv --checkpoint model.ckpt
"""

import logging
import pickle
import urllib.request
from pathlib import Path
from typing import Optional

import click
import torch
import yaml
from rdkit import Chem

from boltz.data.mol import load_canonicals
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import Manifest, Record
from boltz.model.models.affinity_predictor import AffinityPredictor
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule

logger = logging.getLogger(__name__)

# Model checkpoint URLs
BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]

MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"


def get_cache_path() -> Path:
    """Get the cache directory for model checkpoints."""
    cache_dir = Path.home() / ".boltz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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


def download_checkpoint(
    urls: list[str],
    output_path: Path,
    description: str = "checkpoint",
) -> Path:
    """Download checkpoint from URLs with fallback."""
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


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def load_mols() -> dict:
    """Load canonical molecules."""
    cache_dir = get_cache_path()
    mols_path = cache_dir / "mols.tar"
    
    # Download if needed
    if not mols_path.exists():
        logger.info("Downloading canonical molecules...")
        download_checkpoint([MOL_URL], mols_path, "canonical molecules")
    
    # Load molecules
    mols = load_canonicals(mols_path)
    return mols


@click.group()
def cli():
    """Boltz-2 Affinity Predictor CLI."""
    pass


@cli.command()
@click.option("--input", required=True, type=click.Path(exists=True), help="Input YAML config or directory of configs")
@click.option("--output", required=True, type=click.Path(), help="Output directory for results")
@click.option("--checkpoint", required=False, type=click.Path(exists=False), default=None, help="Path to model checkpoint")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
@click.option("--batch-size", default=1, type=int, help="Batch size for inference")
@click.option("--num-workers", default=0, type=int, help="Number of workers for data loading")
@click.option("--recycling-steps", default=0, type=int, help="Number of recycling steps for pairformer")
def predict(input: str, output: str, checkpoint: Optional[str], device: str, batch_size: int, num_workers: int, recycling_steps: int):
    """Predict binding affinity for protein-ligand complexes."""
    # Setup output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download checkpoint if not provided
    if checkpoint is None:
        cache_dir = get_cache_path()
        checkpoint_path = cache_dir / "boltz2_aff.ckpt"
        checkpoint = str(download_checkpoint(BOLTZ2_AFFINITY_URL_WITH_FALLBACK, checkpoint_path, description="Boltz-2 affinity checkpoint"))
    
    logger.info(f"Loading checkpoint from {checkpoint}")
    ckpt = load_checkpoint(checkpoint, device=device)

    # Extract model args from checkpoint
    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", {})
    
    # Validate checkpoint structure
    if not state_dict:
        msg = "Checkpoint does not contain 'state_dict'. Invalid checkpoint file."
        raise ValueError(msg)
    
    logger.debug(f"Checkpoint keys: {list(ckpt.keys())}")
    logger.debug(f"Hyperparameters keys: {list(hparams.keys())}")

    # Initialize model
    logger.info("Initializing affinity predictor")
    
    token_s = hparams.get("token_s", 384)
    token_z = hparams.get("token_z", 128)
    
    inferred_embedder_args = infer_embedder_args_from_state_dict(state_dict)
    embedder_args = hparams.get("embedder_args", {})
    embedder_args = {**embedder_args, **inferred_embedder_args}
    logger.info(f"Inferred embedder args from checkpoint: {inferred_embedder_args}")
    
    # Validate critical parameters were inferred
    required_params = ["atom_s", "atom_z", "atom_feature_dim", "atom_encoder_heads"]
    missing = [p for p in required_params if p not in embedder_args]
    if missing:
        msg = f"Failed to infer required embedder parameters: {missing}. Checkpoint may be incompatible."
        raise ValueError(msg)
    
    affinity_model_args = hparams.get("affinity_model_args", {})
    if "transformer_args" not in affinity_model_args:
        affinity_model_args["transformer_args"] = {}
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
    
    load_result = model.load_state_dict(ckpt["state_dict"], strict=False)
    if load_result.missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {load_result.missing_keys[:5]}...")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {load_result.unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()

    # Load canonical molecules
    logger.info("Loading canonical molecules")
    mols = load_mols()
    
    # Load input YAML files
    logger.info(f"Loading input from {input}")
    input_path = Path(input)
    
    yaml_files = []
    if input_path.is_dir():
        yaml_files = sorted(input_path.glob("*.yaml")) + sorted(input_path.glob("*.yml"))
        logger.info(f"Found {len(yaml_files)} YAML files in {input_path}")
    elif input_path.is_file():
        yaml_files = [input_path]
    
    if not yaml_files:
        msg = f"No YAML files found in {input_path}"
        raise FileNotFoundError(msg)
    
    # Create temporary directories for feature preparation
    temp_dir = Path("/tmp/boltz_affinity_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    msa_dir = temp_dir / "msa"
    msa_dir.mkdir(exist_ok=True)
    targets_dir = temp_dir / "targets"
    targets_dir.mkdir(exist_ok=True)
    mol_dir = temp_dir / "mols"
    mol_dir.mkdir(exist_ok=True)
    
    # Process YAML files and create manifest
    logger.info(f"Processing {len(yaml_files)} YAML files")
    records = []
    
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    
    for yaml_file in yaml_files:
        try:
            logger.info(f"Parsing {yaml_file.name}")
            target = parse_yaml(yaml_file, mols, mol_dir, boltz2=True)
            records.append(target.record)
            logger.debug(f"Successfully parsed {yaml_file.name}: {target.record.id}")
        except Exception as e:
            logger.warning(f"Failed to parse {yaml_file.name}: {e}")
            continue
    
    if not records:
        msg = "No valid YAML files could be parsed"
        raise ValueError(msg)
    
    logger.info(f"Successfully parsed {len(records)} valid configurations")
    
    # Create manifest
    manifest = Manifest(records=records)
    
    # Initialize data module with the prepared data
    logger.info("Initializing data module")
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=targets_dir,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        num_workers=num_workers,
        affinity=True,
    )
    data_module.setup("predict")
    predict_dataloader = data_module.predict_dataloader()
    
    # Run predictions
    logger.info(f"Running predictions ({len(predict_dataloader)} batches)")
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(predict_dataloader):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                coords = batch.get("coords")
                if coords is None:
                    logger.error(f"Batch {batch_idx} missing 'coords' field. Ensure your input includes 3D structures.")
                    continue

                if coords.dim() == 4:
                    coords = coords.squeeze(1)

                out = model(feats=batch, coords=coords, recycling_steps=recycling_steps)

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
                continue

    # Write results
    logger.info(f"Writing results to {output_path}")
    
    if not results:
        logger.warning("No results to write. All batches may have failed or been skipped.")
        import csv
        csv_output = output_path / "affinity_predictions.csv"
        with open(csv_output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["complex_id", "affinity_pred_value", "affinity_probability_binary"])
        return
    
    import csv
    import numpy as np

    csv_output = output_path / "affinity_predictions.csv"
    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["complex_id", "affinity_pred_value", "affinity_probability_binary"])

        for batch_result in results:
            for i, cid in enumerate(batch_result["complex_id"]):
                affinity_value = batch_result["affinity_pred_value"][i, 0]
                affinity_prob = batch_result["affinity_probability_binary"][i, 0]
                writer.writerow([cid, f"{affinity_value:.4f}", f"{affinity_prob:.4f}"])

    logger.info(f"Results saved to {csv_output}. Total predictions: {sum(len(r['complex_id']) for r in results)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cli()
