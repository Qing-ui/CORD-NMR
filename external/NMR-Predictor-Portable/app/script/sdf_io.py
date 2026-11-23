"""Read and write SDF through Python's Unicode-aware file handling."""

from contextlib import contextmanager
from pathlib import Path

from rdkit import Chem


def iter_sdf(path):
    with Path(path).open("rb") as source:
        yield from Chem.ForwardSDMolSupplier(source, removeHs=False)


@contextmanager
def sdf_writer(path):
    with Path(path).open("w", encoding="utf-8", newline="\n") as target:
        writer = Chem.SDWriter(target)
        try:
            yield writer
        finally:
            writer.close()
