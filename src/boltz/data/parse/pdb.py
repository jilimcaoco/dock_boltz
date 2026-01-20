from typing import Optional
from rdkit.Chem.rdchem import Mol
import gemmi
from tempfile import NamedTemporaryFile
from boltz.data.parse.mmcif import parse_mmcif, ParsedStructure

def parse_pdb(
    path: str,
    mols: Optional[dict[str, Mol]] = None,
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    compute_interfaces: bool = True,
    override_first_chain_sequence: Optional[list[str]] = None,
) -> ParsedStructure:
    """Parse a PDB file and delegate to mmCIF parser.
    
    Parameters
    ----------
    path : str
        Path to the PDB file.
    mols : dict, optional
        Dictionary of molecules.
    moldir : str, optional
        Path to molecule directory.
    use_assembly : bool
        Whether to use assembly (default: True).
    compute_interfaces : bool
        Whether to compute interfaces (default: True).
    override_first_chain_sequence : list[str], optional
        Override the sequence for the first polymer chain.
        Useful when the PDB lacks sequence info but you have it from YAML.
    
    Returns
    -------
    ParsedStructure
        The parsed structure.
    """
    with NamedTemporaryFile(suffix=".cif") as tmp_cif_file:
        tmp_cif_path = tmp_cif_file.name
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()

        subchain_counts, subchain_renaming = {}, {}
        for chain in structure[0]:
            subchain_counts[chain.name] = 0
            for res in chain:
                if res.subchain not in subchain_renaming:
                    subchain_renaming[res.subchain] = chain.name + str(subchain_counts[chain.name] + 1)
                    subchain_counts[chain.name] += 1
                res.subchain = subchain_renaming[res.subchain]
        for entity in structure.entities:
            entity.subchains = [subchain_renaming[subchain] for subchain in entity.subchains]

        doc = structure.make_mmcif_document()
        doc.write_file(tmp_cif_path)

        return parse_mmcif(
            path=tmp_cif_path,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=compute_interfaces,
            override_first_chain_sequence=override_first_chain_sequence,
        )