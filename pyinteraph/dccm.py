import argparse

import numpy as np
import itertools

from pyinteraph.core import logger
from pyinteraph.core.residue_coordinate_matrix import ResidueCoordinateMatrix
from pyinteraph.core.validate_parser_file_extension import ArgumentParserFileExtensionValidation


class DCCMAnalyzer:
    def __init__(self, residue_coordinate_matrix: ResidueCoordinateMatrix) -> None:
        self.residue_coordinate_matrix = residue_coordinate_matrix

    def run(self) -> np.ndarray:
        n_residues = self.residue_coordinate_matrix.n_residues
        coordinates_by_residue = self.residue_coordinate_matrix.coordinates_by_residue

        comb = list(itertools.combinations_with_replacement(list(range(0, n_residues)), 2))
        corr = np.zeros(shape=(n_residues, n_residues))
        for i, j in comb:
            corr[i, j] = np.corrcoef(coordinates_by_residue[i, :], coordinates_by_residue[j, :])[1][0]
        return corr

    def to_csv(self) -> None:
        corr = self.run()
        np.savetxt("dccm.csv", corr, comments='', delimiter=',',
                   header=','.join(self.residue_coordinate_matrix.residues_with_atom_number.keys()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atoms', type=str, required=False)
    parser.add_argument('--backbone', action='store_true', required=False)
    parser.add_argument("--ref", help="Reference file",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(
                            (".pdb, .gro, .psf, .top, .crd"), file_name).validate_file_extension(),
                        required=True)
    parser.add_argument("--traj", help="a trajectory file",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(
                            (".trj, .pdb, .xtc, .dcd"), file_name).validate_file_extension(),
                        required=True)
    args = parser.parse_args()

    backbone = False
    if (args.atoms and args.backbone) or not args.atoms:
        logger.warning(f"Backbone atoms will be utilized to compute dccm.")
        backbone = True

    residue_coordinate_matrix = ResidueCoordinateMatrix(args.ref, args.traj, args.atoms, backbone)
    DCCMAnalyzer(residue_coordinate_matrix).to_csv()


if __name__ == "__main__":
    main()
