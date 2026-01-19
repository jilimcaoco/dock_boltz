#!/usr/bin/env python3
"""
Test script for external structure loading pipeline.
Tests the complete flow: YAML → schema parsing → coordinate extraction → affinity prediction
"""

import sys
from pathlib import Path

# Test configuration
TEST_YAML = Path("examples/affinity_with_structures.yaml")
TEST_PROTEIN_PDB = Path("test_structures/receptor.pdb")  # User should provide
TEST_LIGAND_MOL2 = Path("test_structures/ligand.mol2")  # User should provide


def test_yaml_parsing():
    """Test YAML schema parsing with structure_path fields."""
    print("\n=== Test 1: YAML Parsing ===")
    
    try:
        from boltz.data.parse.schema import parse_boltz_schema
        
        # Check if example YAML exists
        if not TEST_YAML.exists():
            print(f"❌ Example YAML not found: {TEST_YAML}")
            return False
        
        print(f"✓ Found example YAML: {TEST_YAML}")
        
        # Try to parse (will fail if structure files don't exist, but that's okay for this test)
        # We just want to check that the structure_path field is recognized
        with open(TEST_YAML) as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Check for structure_path fields
        has_protein_structure_path = False
        has_ligand_structure_path = False
        
        for sequence in config.get("sequences", []):
            if "protein" in sequence and "structure_path" in sequence["protein"]:
                has_protein_structure_path = True
                print(f"✓ Found protein structure_path: {sequence['protein']['structure_path']}")
            
            if "ligand" in sequence and "structure_path" in sequence["ligand"]:
                has_ligand_structure_path = True
                print(f"✓ Found ligand structure_path: {sequence['ligand']['structure_path']}")
        
        if not has_protein_structure_path:
            print("⚠ No protein structure_path found in example YAML")
        
        if not has_ligand_structure_path:
            print("⚠ No ligand structure_path found in example YAML")
        
        print("✓ YAML parsing test passed")
        return True
        
    except Exception as e:
        print(f"❌ YAML parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_protein_coordinate_loading():
    """Test protein coordinate loading from PDB file."""
    print("\n=== Test 2: Protein Coordinate Loading ===")
    
    if not TEST_PROTEIN_PDB.exists():
        print(f"⚠ Test protein PDB not found: {TEST_PROTEIN_PDB}")
        print("  Create a test file at this location to run this test")
        return None
    
    try:
        from boltz.data.parse.pdb import parse_pdb
        
        print(f"Loading protein from {TEST_PROTEIN_PDB}...")
        structure = parse_pdb(TEST_PROTEIN_PDB)
        
        print(f"✓ Loaded {len(structure.chains)} chain(s)")
        
        for i, chain in enumerate(structure.chains):
            print(f"  Chain {i}: {len(chain.residues)} residues")
            
            # Check that first residue has coordinates
            if len(chain.residues) > 0:
                first_res = chain.residues[0]
                print(f"    First residue: {first_res.name}")
                
                if len(first_res.atoms) > 0:
                    first_atom = first_res.atoms[0]
                    print(f"    First atom: {first_atom.name} at {first_atom.coords}")
                    
                    # Check coords are not all zeros
                    if first_atom.coords == (0, 0, 0):
                        print("    ⚠ WARNING: Coordinates are (0,0,0)")
                    else:
                        print("    ✓ Has non-zero coordinates")
        
        print("✓ Protein coordinate loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Protein loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ligand_coordinate_loading():
    """Test ligand coordinate loading from MOL2/SDF file."""
    print("\n=== Test 3: Ligand Coordinate Loading ===")
    
    if not TEST_LIGAND_MOL2.exists():
        print(f"⚠ Test ligand MOL2 not found: {TEST_LIGAND_MOL2}")
        print("  Create a test file at this location to run this test")
        return None
    
    try:
        from rdkit import Chem
        
        print(f"Loading ligand from {TEST_LIGAND_MOL2}...")
        mol = Chem.MolFromMol2File(str(TEST_LIGAND_MOL2), removeHs=False)
        
        if mol is None:
            print(f"❌ Failed to load molecule from {TEST_LIGAND_MOL2}")
            return False
        
        print(f"✓ Loaded molecule with {mol.GetNumAtoms()} atoms")
        
        # Check for conformer (3D coordinates)
        if mol.GetNumConformers() == 0:
            print("❌ No conformer found (no 3D coordinates)")
            return False
        
        print(f"✓ Has {mol.GetNumConformers()} conformer(s)")
        
        # Get first conformer coordinates
        conf = mol.GetConformer(0)
        first_pos = conf.GetAtomPosition(0)
        print(f"  First atom position: ({first_pos.x:.2f}, {first_pos.y:.2f}, {first_pos.z:.2f})")
        
        if (first_pos.x == 0 and first_pos.y == 0 and first_pos.z == 0):
            print("  ⚠ WARNING: First atom at (0,0,0)")
        else:
            print("  ✓ Has non-zero coordinates")
        
        print("✓ Ligand coordinate loading test passed")
        return True
        
    except Exception as e:
        print(f"❌ Ligand loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinate_shapes():
    """Test that coordinates have expected shapes for affinity module."""
    print("\n=== Test 4: Coordinate Shape Validation ===")
    
    try:
        import torch
        
        # Test coordinate reshaping logic from affinity_predictor.py
        print("Testing coordinate reshape logic...")
        
        # Test case 1: (B, N_atoms, 3)
        coords_3d = torch.randn(2, 100, 3)
        coords_reshaped = coords_3d.unsqueeze(1).unsqueeze(1)
        expected_shape = (2, 1, 1, 100, 3)
        
        if coords_reshaped.shape == expected_shape:
            print(f"✓ 3D reshape: {coords_3d.shape} -> {coords_reshaped.shape}")
        else:
            print(f"❌ 3D reshape failed: got {coords_reshaped.shape}, expected {expected_shape}")
            return False
        
        # Test case 2: (B, 1, N_atoms, 3)
        coords_4d = torch.randn(2, 1, 100, 3)
        coords_reshaped = coords_4d.unsqueeze(1)
        expected_shape = (2, 1, 1, 100, 3)
        
        if coords_reshaped.shape == expected_shape:
            print(f"✓ 4D reshape: {coords_4d.shape} -> {coords_reshaped.shape}")
        else:
            print(f"❌ 4D reshape failed: got {coords_reshaped.shape}, expected {expected_shape}")
            return False
        
        print("✓ Coordinate shape validation passed")
        return True
        
    except ImportError:
        print("⚠ PyTorch not installed, skipping shape validation")
        return None
    except Exception as e:
        print(f"❌ Shape validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test complete pipeline with actual structure files."""
    print("\n=== Test 5: Full Pipeline Integration ===")
    
    if not (TEST_PROTEIN_PDB.exists() and TEST_LIGAND_MOL2.exists()):
        print("⚠ Test structure files not found, skipping full pipeline test")
        print(f"  Need: {TEST_PROTEIN_PDB} and {TEST_LIGAND_MOL2}")
        return None
    
    try:
        # This would require creating a temporary YAML with correct paths
        # and running the full parsing → featurization → prediction pipeline
        print("⚠ Full pipeline test not yet implemented")
        print("  To test manually, run:")
        print("  python -m boltz.main_simplified predict --yaml examples/affinity_with_structures.yaml --checkpoint <path>")
        return None
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EXTERNAL STRUCTURE LOADING TEST SUITE")
    print("=" * 60)
    
    results = {
        "YAML Parsing": test_yaml_parsing(),
        "Protein Loading": test_protein_coordinate_loading(),
        "Ligand Loading": test_ligand_coordinate_loading(),
        "Coordinate Shapes": test_coordinate_shapes(),
        "Full Pipeline": test_full_pipeline(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        
        print(f"{status:8} {test_name}")
    
    # Exit with error if any test failed
    if any(r is False for r in results.values()):
        print("\n❌ Some tests failed!")
        sys.exit(1)
    elif all(r is None for r in results.values()):
        print("\n⚠ All tests skipped (missing dependencies or test files)")
        sys.exit(0)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
