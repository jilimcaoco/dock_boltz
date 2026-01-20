from pathlib import Path

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target
import logging
import sys
import traceback
import linecache

logger = logging.getLogger(__name__)

def parse_yaml(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    templates:
        - path: /path/to/template.pdb
          ids: [A] # optional, specify which chains to template

    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as fp:
        schema = yaml.safe_load(fp)
    name = path.stem
    try:
        return parse_boltz_schema(name, schema, ccd, mol_dir, boltz_2=boltz2)
    except IndexError as e:
        # Find the last traceback frame where the IndexError occurred
        tb = e.__traceback__
        while tb.tb_next is not None:
            tb = tb.tb_next
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        filename = frame.f_code.co_filename
        code_line = linecache.getline(filename, lineno).strip()

        # Collect small snapshot of locals
        small_locals = {}
        for k, v in frame.f_locals.items():
            try:
                small_locals[k] = repr(v)[:500]
            except Exception:
                small_locals[k] = f"<unrepr: {type(v).__name__}>"

        logger.error(
            "IndexError while parsing YAML %s: %s at %s:%d -> %s\nLocals snapshot: %s",
            path,
            e,
            filename,
            lineno,
            code_line,
            small_locals,
        )

        # Raise a new error with diagnostic info for the user
        raise IndexError(
            f"IndexError while parsing {path}: {e} at {filename}:{lineno} -> {code_line}. "
            f"Locals keys: {list(small_locals.keys())}"
        ) from e
