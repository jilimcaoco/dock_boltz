#!/usr/bin/env python3
"""
Generate YAML configs for batch affinity scoring of docked poses.

Usage:
    # Provide sequence explicitly
    python scripts/generate_affinity_yaml.py \
        --protein receptor.pdb \
        --protein-seq MKFLKF... \
        --ligand "c1ccccc1" \
        --poses-dir ./docked_poses/ \
        --output batch_affinity.yaml

    # Auto-extract sequence from PDB
    python scripts/generate_affinity_yaml.py \
        --protein receptor.pdb \
        --ligand "c1ccccc1" \
        --poses-dir ./docked_poses/ \
        --output batch_affinity.yaml

    # Auto-extract both sequence and SMILES
    python scripts/generate_affinity_yaml.py \
        --protein receptor.pdb \
        --poses-dir ./docked_poses/ \
        --auto-smiles \
        --output batch_affinity.yaml

Note: Requires BioPython for automatic sequence extraction:
    pip install biopython
"""

import sys
from pathlib import Path
from typing import Optional
import argparse

try:
    import yaml
    from rdkit import Chem
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Install with: pip install pyyaml rdkit-pypi")
    sys.exit(1)

try:
    from Bio import PDB
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# Standard amino acid one-letter codes
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',  # Selenomethionine
}


def get_sequence_from_pdb(pdb_file: Path) -> Optional[str]:
    """Extract protein sequence from PDB file."""
    if not HAS_BIOPYTHON:
        print("Warning: BioPython not available, cannot auto-extract sequence")
        print("Install with: pip install biopython")
        return None
    
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', str(pdb_file))
        
        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in AA_MAP:
                        sequence.append(AA_MAP[resname])
        
        if not sequence:
            print(f"Warning: Could not extract sequence from {pdb_file}")
            return None
        
        seq_str = "".join(sequence)
        print(f"Extracted sequence from {pdb_file.name}: {len(seq_str)} residues")
        return seq_str
    except Exception as e:
        print(f"Warning: Error extracting sequence from {pdb_file}: {e}")
        return None


def get_molecule_smiles(mol_file: Path) -> Optional[str]:
    """Extract SMILES from molecule file."""
    try:
        if mol_file.suffix.lower() == ".mol2":
            mol = Chem.MolFromMol2File(str(mol_file), removeHs=False)
        elif mol_file.suffix.lower() in [".sdf", ".sd"]:
            supplier = Chem.SDMolSupplier(str(mol_file), removeHs=False)
            mol = supplier[0] if supplier else None
        elif mol_file.suffix.lower() == ".pdb":
            mol = Chem.MolFromPDBFile(str(mol_file), removeHs=False)
        else:
            return None
        
        if mol is None:
            return None
        
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Warning: Could not extract SMILES from {mol_file}: {e}")
        return None


def validate_structure_file(filepath: Path) -> bool:
    """Check that structure file exists and has coordinates."""
    if not filepath.exists():
        print(f"  ✗ File not found: {filepath}")
        return False
    
    try:
        # Try to load and check for coordinates
        if filepath.suffix.lower() == ".mol2":
            mol = Chem.MolFromMol2File(str(filepath), removeHs=False)
        elif filepath.suffix.lower() in [".sdf", ".sd"]:
            supplier = Chem.SDMolSupplier(str(filepath), removeHs=False)
            mol = supplier[0] if supplier else None
        elif filepath.suffix.lower() == ".pdb":
            mol = Chem.MolFromPDBFile(str(filepath), removeHs=False)
        else:
            print(f"  ✗ Unsupported format: {filepath.suffix}")
            return False
        
        if mol is None:
            print(f"  ✗ Could not load molecule: {filepath}")
            return False
        
        if mol.GetNumConformers() == 0:
            print(f"  ✗ No 3D coordinates: {filepath}")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error validating {filepath}: {e}")
        return False


def generate_yaml_single_pose(
    protein_pdb: Path,
    protein_seq: str,
    ligand_smiles: str,
    pose_file: Path,
    pose_idx: int = 0,
) -> dict:
    """Generate YAML sequence entry for a single pose."""
    return {
        "protein": {
            "id": "A",
            "sequence": protein_seq,
            "structure_path": str(protein_pdb),
        },
        "ligand": {
            "id": "B",
            "smiles": ligand_smiles,
            "structure_path": str(pose_file),
        },
    }


def generate_batch_yaml(
    protein_pdb: Path,
    protein_seq: str,
    ligand_smiles: str,
    poses_dir: Path,
    output_yaml: Path,
    file_pattern: str = "*.mol2",
    validate: bool = True,
) -> int:
    """
    Generate YAML config for batch scoring of poses.
    
    Parameters
    ----------
    protein_pdb : Path
        Path to protein structure file
    protein_seq : str
        Protein sequence
    ligand_smiles : str
        Ligand SMILES string
    poses_dir : Path
        Directory containing docked poses
    output_yaml : Path
        Output YAML file path
    file_pattern : str
        Glob pattern for pose files (default: "*.mol2")
    validate : bool
        Whether to validate each pose file
        
    Returns
    -------
    int
        Number of poses added to YAML
    """
    poses_dir = Path(poses_dir)
    
    # Validate inputs
    if not protein_pdb.exists():
        print(f"Error: Protein file not found: {protein_pdb}")
        return 0
    
    if not poses_dir.is_dir():
        print(f"Error: Poses directory not found: {poses_dir}")
        return 0
    
    # Find pose files
    pose_files = sorted(poses_dir.glob(file_pattern))
    
    if not pose_files:
        print(f"Error: No pose files matching '{file_pattern}' found in {poses_dir}")
        return 0
    
    print(f"Found {len(pose_files)} pose file(s)")
    
    # Validate each pose
    if validate:
        print("Validating pose files...")
        valid_poses = []
        for pose_file in pose_files:
            if validate_structure_file(pose_file):
                valid_poses.append(pose_file)
            else:
                print(f"  Skipping: {pose_file.name}")
        
        pose_files = valid_poses
        print(f"Valid poses: {len(pose_files)}/{len(pose_files)}")
    
    if not pose_files:
        print("Error: No valid pose files")
        return 0
    
    # Generate YAML
    yaml_config = {
        "version": 1,
        "sequences": [],
    }
    
    for pose_file in pose_files:
        sequence = generate_yaml_single_pose(
            protein_pdb=protein_pdb,
            protein_seq=protein_seq,
            ligand_smiles=ligand_smiles,
            pose_file=pose_file,
        )
        yaml_config["sequences"].append(sequence)
    
    # Write YAML
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_yaml, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nGenerated YAML: {output_yaml}")
    print(f"Total sequences: {len(yaml_config['sequences'])}")
    
    return len(yaml_config["sequences"])


def generate_individual_yamls(
    protein_pdb: Path,
    protein_seq: str,
    ligand_smiles: str,
    poses_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.mol2",
    validate: bool = True,
) -> int:
    """
    Generate individual YAML files for each pose.
    One YAML file per compound using the filename as the compound name.
    Useful for parallel processing.
    
    Parameters
    ----------
    protein_pdb : Path
        Path to protein structure file
    protein_seq : str
        Protein sequence
    ligand_smiles : str
        Ligand SMILES string
    poses_dir : Path
        Directory containing docked poses
    output_dir : Path
        Output directory for YAML files
    file_pattern : str
        Glob pattern for pose files (default: "*.mol2")
    validate : bool
        Whether to validate each pose file
        
    Returns
    -------
    int
        Number of YAML files generated
    """
    poses_dir = Path(poses_dir)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not protein_pdb.exists():
        print(f"Error: Protein file not found: {protein_pdb}")
        return 0
    
    if not poses_dir.is_dir():
        print(f"Error: Poses directory not found: {poses_dir}")
        return 0
    
    # Find pose files
    pose_files = sorted(poses_dir.glob(file_pattern))
    
    if not pose_files:
        print(f"Error: No pose files matching '{file_pattern}' found in {poses_dir}")
        return 0
    
    print(f"Found {len(pose_files)} pose file(s)")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate YAML for each pose
    count = 0
    for pose_file in pose_files:
        if validate:
            if not validate_structure_file(pose_file):
                print(f"  Skipping: {pose_file.name}")
                continue
        
        # Create YAML config - wrap sequence as single list item
        sequence = generate_yaml_single_pose(
            protein_pdb=protein_pdb,
            protein_seq=protein_seq,
            ligand_smiles=ligand_smiles,
            pose_file=pose_file,
        )
        yaml_config = {
            "version": 1,
            "sequences": [sequence],  # Single dict with protein and ligand keys
        }
        
        # Write YAML using filename (without extension) as compound name
        compound_name = pose_file.stem
        yaml_filename = output_dir / f"{compound_name}.yaml"
        with open(yaml_filename, "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ {yaml_filename.name}")
        count += 1
    
    print(f"\nGenerated {count} YAML file(s) in {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML configs for batch affinity scoring of docked poses"
    )
    
    parser.add_argument(
        "--protein",
        type=Path,
        required=True,
        help="Path to protein structure file (PDB/mmCIF)",
    )
    
    parser.add_argument(
        "--protein-seq",
        type=str,
        help="Protein sequence (FASTA). If not provided, will be auto-extracted from PDB (requires BioPython)",
    )
    
    parser.add_argument(
        "--ligand",
        type=str,
        help="Ligand SMILES string",
    )
    
    parser.add_argument(
        "--poses-dir",
        type=Path,
        required=True,
        help="Directory containing docked pose files",
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mol2",
        help="Glob pattern for pose files (default: '*.mol2'). Use '*.sdf' for SDF, '*.pdb' for PDB",
    )
    
    parser.add_argument(
        "--auto-smiles",
        action="store_true",
        help="Automatically extract SMILES from first pose file (requires --ligand not set)",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output YAML file (single file with all poses). Default: batch_affinity.yaml",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for individual YAML files (one per pose)",
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of structure files",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ligand and not args.auto_smiles:
        print("Error: Must provide --ligand or --auto-smiles")
        parser.print_help()
        sys.exit(1)
    
    if args.ligand and args.auto_smiles:
        print("Error: Cannot use both --ligand and --auto-smiles")
        sys.exit(1)
    
    # Get ligand SMILES
    if args.auto_smiles:
        print("Auto-extracting SMILES from first pose...")
        first_pose = sorted(args.poses_dir.glob(args.pattern))[0]
        ligand_smiles = get_molecule_smiles(first_pose)
        if not ligand_smiles:
            print(f"Error: Could not extract SMILES from {first_pose}")
            sys.exit(1)
        print(f"  Extracted: {ligand_smiles}")
    else:
        ligand_smiles = args.ligand
    
    # Get protein sequence
    protein_seq = args.protein_seq
    if not protein_seq:
        print("Auto-extracting protein sequence from PDB...")
        protein_seq = get_sequence_from_pdb(args.protein)
        if not protein_seq:
            print("Error: Could not extract sequence from PDB and --protein-seq not provided")
            print("Please either:")
            print("  1. Provide --protein-seq explicitly")
            print("  2. Install BioPython: pip install biopython")
            sys.exit(1)
        print(f"  Extracted: {protein_seq[:50]}..." if len(protein_seq) > 50 else f"  Extracted: {protein_seq}")
    
    validate = not args.no_validate
    
    # Generate YAML(s)
    if args.output_dir:
        count = generate_individual_yamls(
            protein_pdb=args.protein,
            protein_seq=protein_seq,
            ligand_smiles=ligand_smiles,
            poses_dir=args.poses_dir,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            validate=validate,
        )
    else:
        output_yaml = args.output or Path("batch_affinity.yaml")
        count = generate_batch_yaml(
            protein_pdb=args.protein,
            protein_seq=protein_seq,
            ligand_smiles=ligand_smiles,
            poses_dir=args.poses_dir,
            output_yaml=output_yaml,
            file_pattern=args.pattern,
            validate=validate,
        )
    
    if count == 0:
        print("Error: No poses added to YAML")
        sys.exit(1)
    
    print(f"\n✓ Success! Ready to run affinity predictions")
    if args.output:
        print(f"\n  python -m boltz.main_simplified predict \\")
        print(f"      --yaml {args.output} \\")
        print(f"      --checkpoint checkpoint.ckpt \\")
        print(f"      --output results/")
    elif args.output_dir:
        print(f"\n  # Process each YAML with your favorite batch tool")
        print(f"  for yaml in {args.output_dir}/*.yaml; do")
        print(f"    python -m boltz.main_simplified predict \\")
        print(f"        --yaml $yaml \\")
        print(f"        --checkpoint checkpoint.ckpt \\")
        print(f"        --output results/")
        print(f"  done")


if __name__ == "__main__":
    main()
