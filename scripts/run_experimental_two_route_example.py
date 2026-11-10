"""Generate a synthetic input and run both experimental sampler modes."""

import argparse
from pathlib import Path

import numpy as np

from pigean.experimental_two_route import main as run_sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New directory for the example input and fits")
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)

    # Two signals share G2; G4 has an annotation but no nearby signal.
    all_genes = np.array([[1.0, 0.3, -0.5, 0.8]])
    source = out / "input.npz"
    np.savez_compressed(
        source, mat=all_genes[:, :3].copy(), all_gene_mat=all_genes,
        keep=[0, 1, 2], panel=[0, 1, 2, -1], ids=[0, 1, 1, 2],
        ptr=[0, 2, 4], weights=[0.8, 0.2, 0.65, 0.35],
        initial=[0.0], pi=[0.5], sd=[1.25],
    )
    print("sampler\tgene\tSNP-only assignment\tfitted assignment\tindirect\teither")
    for label, fraction in (("centered", "0"), ("interwoven", "0.75")):
        result = out / f"{label}.npz"
        run_sampler([
            str(source), str(result), "--random-seed", "7", "--steps", "12000",
            "--burn", "2000", "--joint-move-probability", fraction,
        ])
        with np.load(result) as f:
            for i, row in enumerate(zip(
                f["direct_without_annotations"], f["direct_with_annotations"],
                f["indirect"], f["either"],
            ), 1):
                print("\t".join((label, f"G{i}", *(f"{x:.4f}" for x in row))))


if __name__ == "__main__":
    main()
