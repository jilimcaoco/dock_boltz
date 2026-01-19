"""
Simplified Boltz-2 for affinity prediction with external structures.

This model computes evoformer embeddings from sequences and predicts
binding affinity from external 3D structures.
"""

from typing import Any, Optional

import torch
from torch import Tensor, nn

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.affinity import AffinityModule
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.modules.trunkv2 import (
    ContactConditioning,
    InputEmbedder,
    MSAModule,
)


class AffinityPredictor(nn.Module):
    """Simplified Boltz-2 for affinity prediction from external structures."""

    def __init__(
        self,
        token_s: int = 384,
        token_z: int = 128,
        embedder_args: Optional[dict[str, Any]] = None,
        msa_args: Optional[dict[str, Any]] = None,
        pairformer_args: Optional[dict[str, Any]] = None,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_ensemble: bool = False,
        affinity_model_args1: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        affinity_mw_correction: bool = True,
        conditioning_cutoff_min: float = 4.0,
        conditioning_cutoff_max: float = 20.0,
        cyclic_pos_enc: bool = False,
        bond_type_feature: bool = False,
    ) -> None:
        """Initialize the affinity predictor.

        Parameters
        ----------
        token_s : int
            Token embedding dimension
        token_z : int
            Token pairwise embedding dimension
        embedder_args : dict
            Arguments for InputEmbedder
        msa_args : dict
            Arguments for MSAModule
        pairformer_args : dict
            Arguments for PairformerModule
        affinity_model_args : dict
            Arguments for AffinityModule
        affinity_ensemble : bool
            Whether to use ensemble of two affinity models
        affinity_mw_correction : bool
            Whether to apply molecular weight correction

        """
        super().__init__()

        self.token_s = token_s
        self.token_z = token_z
        self.affinity_prediction = True
        self.affinity_ensemble = affinity_ensemble
        self.affinity_mw_correction = affinity_mw_correction

        # Input embeddings
        embedder_args = embedder_args or {}
        self.input_embedder = InputEmbedder(
            token_s=token_s,
            token_z=token_z,
            **embedder_args,
        )

        # Sequence embeddings
        self.s_init = nn.Linear(token_s, token_s, bias=False)

        # Pairwise embeddings
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)

        # Relative position encoding
        self.rel_pos = RelativePositionEncoder(
            token_z, cyclic_pos_enc=cyclic_pos_enc
        )

        # Token bonds
        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.bond_type_feature = bond_type_feature
        if bond_type_feature:
            self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)

        # Contact conditioning
        self.contact_conditioning = ContactConditioning(
            token_z=token_z,
            cutoff_min=conditioning_cutoff_min,
            cutoff_max=conditioning_cutoff_max,
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # MSA module
        msa_args = msa_args or {}
        self.msa_module = MSAModule(
            token_z=token_z,
            token_s=token_s,
            **msa_args,
        )

        # Pairformer module (evoformer)
        pairformer_args = pairformer_args or {}
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)

        # Affinity module(s)
        if self.affinity_ensemble:
            affinity_model_args1 = affinity_model_args1 or {}
            affinity_model_args2 = affinity_model_args2 or {}
            self.affinity_module1 = AffinityModule(
                token_s, token_z, **affinity_model_args1
            )
            self.affinity_module2 = AffinityModule(
                token_s, token_z, **affinity_model_args2
            )
        else:
            affinity_model_args = affinity_model_args or {}
            self.affinity_module = AffinityModule(
                token_s, token_z, **affinity_model_args
            )

    def forward(
        self,
        feats: dict[str, Tensor],
        coords: Tensor,
        recycling_steps: int = 0,
    ) -> dict[str, Tensor]:
        """Run affinity prediction.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features (sequence, MSA, structure info)
        coords : Tensor
            Predicted 3D coordinates, shape (B, N_atoms, 3)
        recycling_steps : int
            Number of recycling steps for pairformer

        Returns
        -------
        dict[str, Tensor]
            Affinity predictions

        """
        # Input embeddings
        s_inputs = self.input_embedder(feats)

        # Initialize embeddings
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        relative_position_encoding = self.rel_pos(feats)
        z_init = z_init + relative_position_encoding
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        # Perform pairformer rounds with recycling
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        for i in range(recycling_steps + 1):
            # Apply recycling
            s = s_init + self.s_norm(s) @ self.s_init.weight.T
            z = z_init + self.z_norm(z) @ self.z_init_1.weight.T[:, :, None]

            # Run MSA module
            z = z + self.msa_module(z, s_inputs, feats)

            # Run pairformer
            s, z = self.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask
            )

        # Prepare affinity prediction
        pad_token_mask = feats["token_pad_mask"]
        rec_mask = (feats["mol_type"] == 0) * pad_token_mask
        lig_mask = feats["affinity_token_mask"].to(torch.bool) * pad_token_mask

        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )

        z_affinity = z * cross_pair_mask[:, :, :, None]

        # Reshape coordinates for affinity module
        # Expected shape: (B, 1, 1, N_atoms, 3) or just (B, N_atoms, 3)
        if len(coords.shape) == 3:
            # (B, N_atoms, 3) -> (B, 1, 1, N_atoms, 3)
            coords_affinity = coords.unsqueeze(1).unsqueeze(1)
        elif len(coords.shape) == 4:
            # (B, 1, N_atoms, 3) -> (B, 1, 1, N_atoms, 3)
            coords_affinity = coords.unsqueeze(1)
        else:
            # Already correct shape
            coords_affinity = coords

        # Run affinity module
        s_inputs_affinity = self.input_embedder(feats, affinity=True)

        dict_out = {}
        if self.affinity_ensemble:
            dict_out_aff1 = self.affinity_module1(
                s_inputs=s_inputs_affinity.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
            )
            dict_out_aff1["affinity_probability_binary"] = torch.sigmoid(
                dict_out_aff1["affinity_logits_binary"]
            )

            dict_out_aff2 = self.affinity_module2(
                s_inputs=s_inputs_affinity.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
            )
            dict_out_aff2["affinity_probability_binary"] = torch.sigmoid(
                dict_out_aff2["affinity_logits_binary"]
            )

            # Average ensemble
            dict_out["affinity_pred_value"] = (
                dict_out_aff1["affinity_pred_value"]
                + dict_out_aff2["affinity_pred_value"]
            ) / 2
            dict_out["affinity_probability_binary"] = (
                dict_out_aff1["affinity_probability_binary"]
                + dict_out_aff2["affinity_probability_binary"]
            ) / 2

            # Apply MW correction if enabled
            if self.affinity_mw_correction:
                model_coef = 1.03525938
                mw_coef = -0.59992683
                bias = 2.83288489
                mw = feats["affinity_mw"][0] ** 0.3
                dict_out["affinity_pred_value"] = (
                    model_coef * dict_out["affinity_pred_value"]
                    + mw_coef * mw
                    + bias
                )
        else:
            dict_out_aff = self.affinity_module(
                s_inputs=s_inputs_affinity.detach(),
                z=z_affinity.detach(),
                x_pred=coords_affinity,
                feats=feats,
                multiplicity=1,
            )
            dict_out["affinity_pred_value"] = dict_out_aff["affinity_pred_value"]
            dict_out["affinity_probability_binary"] = torch.sigmoid(
                dict_out_aff["affinity_logits_binary"]
            )

        return dict_out
