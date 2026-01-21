#!/usr/bin/env python3
"""
Normalize PDB files by converting non-standard residue names to standard ones.

This preprocessing script converts non-standard amino acid residues to their
standard equivalents to avoid alignment issues during parsing.

Normalizations:
- CYX (disulfide-bonded cysteine) → CYS
- MSE (selenomethionine) → MET
- HID, HIE, HIP (histidine variants) → HIS
- SEP (phosphoserine) → SER
- TPO (phosphothreonine) → THR
- PTR (phosphotyrosine) → TYR

Usage:
    python scripts/normalize_pdb.py input.pdb -o output.pdb
    python scripts/normalize_pdb.py input.pdb --inplace
    python scripts/normalize_pdb.py receptors/AA2AR_rec.pdb -o receptors/AA2AR_rec_normalized.pdb
"""

import argparse
import sys
from pathlib import Path

try:
    import gemmi
except ImportError:
    print("Error: gemmi is required. Install with: pip install gemmi")
    sys.exit(1)


# Residue normalization mapping
RESIDUE_NORMALIZATION = {
    "CYX": "CYS",  # Disulfide-bonded cysteine
    "MSE": "MET",  # Selenomethionine
    "HID": "HIS",  # Histidine delta-protonated
    "HIE": "HIS",  # Histidine epsilon-protonated
    "HIP": "HIS",  # Histidine doubly protonated
    "SEP": "SER",  # Phosphoserine
    "TPO": "THR",  # Phosphothreonine
    "PTR": "TYR",  # Phosphotyrosine
}


def normalize_pdb(input_path: Path, output_path: Path, verbose: bool = True) -> dict:
    """
    Normalize residue names in a PDB file.
    
    Parameters
    ----------
    input_path : Path
        Input PDB file path
    output_path : Path
        Output PDB file path
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Statistics about normalizations performed
    """
    if verbose:
        print(f"Loading structure from {input_path}")
    
    # Load structure
    structure = gemmi.read_structure(str(input_path))
    
    # Track statistics
    stats = {residue: 0 for residue in RESIDUE_NORMALIZATION.keys()}
    total_residues = 0
    total_chains = 0
    
    # Process all models, chains, and residues
    for model in structure:
        for chain in model:
            total_chains += 1
            for residue in chain:
                total_residues += 1
                res_name = residue.name.strip()
                
                # Check if normalization needed
                if res_name in RESIDUE_NORMALIZATION:
                    normalized_name = RESIDUE_NORMALIZATION[res_name]
                    if verbose:
                        print(f"  Chain {chain.name}, residue {residue.seqid.num}: {res_name} → {normalized_name}")
                    
                    # Update residue name
                    residue.name = normalized_name
                    stats[res_name] += 1
    
    # Write normalized structure
    if verbose:
        print(f"\nWriting normalized structure to {output_path}")
    structure.write_pdb(str(output_path))
    
    # Add summary stats
    stats['total_residues'] = total_residues
    stats['total_chains'] = total_chains
    stats['total_normalized'] = sum(v for k, v in stats.items() if k in RESIDUE_NORMALIZATION)
    
    return stats


def print_summary(stats: dict):
    """Print normalization summary."""
    print("\n" + "="*80)
    print("NORMALIZATION SUMMARY")
    print("="*80)
    print(f"Total chains:   {stats['total_chains']}")
    print(f"Total residues: {stats['total_residues']}")
    print(f"Total normalized: {stats['total_normalized']}")
    
    if stats['total_normalized'] > 0:
        print("\nNormalizations performed:")
        for res_type, count in stats.items():
            if res_type in RESIDUE_NORMALIZATION and count > 0:
                normalized = RESIDUE_NORMALIZATION[res_type]
                print(f"  {res_type} → {normalized}: {count} residues")
    else:
        print("\n✓ No normalizations needed - all residues are standard")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize non-standard residue names in PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize to a new file
  python scripts/normalize_pdb.py receptor.pdb -o receptor_normalized.pdb
  
  # Normalize in place (overwrites original)
  python scripts/normalize_pdb.py receptor.pdb --inplace
  
  # Normalize with default output name
  python scripts/normalize_pdb.py receptor.pdb
  
Residue normalizations:
  CYX → CYS (disulfide-bonded cysteine)
  MSE → MET (selenomethionine)
  HID/HIE/HIP → HIS (histidine variants)
  SEP → SER (phosphoserine)
  TPO → THR (phosphothreonine)
  PTR → TYR (phosphotyrosine)
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input PDB file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output PDB file path (default: <input_name>_normalized.pdb)"
    )
    
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Normalize in place (overwrites input file)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.suffix.lower() in ['.pdb', '.ent']:
        print(f"Warning: File does not have .pdb extension: {input_path}")
    
    # Determine output path
    if args.inplace:
        output_path = input_path
        if not args.quiet:
            print(f"Normalizing {input_path} in place...")
    elif args.output:
        output_path = Path(args.output)
    else:
        # Default: add _normalized suffix
        output_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize
    try:
        stats = normalize_pdb(input_path, output_path, verbose=not args.quiet)
        
        if not args.quiet:
            print_summary(stats)
            print(f"\n✓ Success! Normalized structure written to: {output_path}")
        
    except Exception as e:
        print(f"Error normalizing PDB file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
