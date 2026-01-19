"""
Simplified CLI for affinity prediction with external structures.

Usage:
    python -m boltz.main predict --input config.yaml --output results.csv --checkpoint model.ckpt
"""

import logging
from pathlib import Path
from typing import Optional

import click
import torch
import yaml

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.model.models.affinity_predictor import AffinityPredictor

logger = logging.getLogger(__name__)


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
    required=True,
    type=click.Path(exists=True),
    help="Path to model checkpoint",
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
    checkpoint: str,
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
    logger.info(f"Loading checkpoint from {checkpoint}")
    ckpt = load_checkpoint(checkpoint, device=device)

    # Extract model args from checkpoint
    hparams = ckpt.get("hyper_parameters", {})

    # Initialize model
    logger.info("Initializing affinity predictor")
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
    model.load_state_dict(ckpt["state_dict"], strict=False)
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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get 3D coordinates from batch
            coords = batch.get("coords")
            if coords is None:
                logger.error("Batch missing 'coords' field. Ensure your input includes 3D structures.")
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

    # Write results
    logger.info(f"Writing results to {output}")
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

    logger.info(f"Results saved to {output}")


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
