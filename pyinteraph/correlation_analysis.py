import argparse
import warnings

from pyinteraph.core import logger
from pyinteraph.correlation.dccm import DCCMAnalyzer
from pyinteraph.correlation.lmi import LMIAnalyzer
from pyinteraph.correlation.residue_coordinate_matrix import ResidueCoordinateMatrix
from pyinteraph.core.validate_parser_file_extension import ArgumentParserFileExtensionValidation

warnings.filterwarnings("ignore")

def run_dccm(args, backbone):
    residue_coordinate_matrix = ResidueCoordinateMatrix(args.ref, args.traj, args.atoms, backbone)
    DCCMAnalyzer(residue_coordinate_matrix, threshold=args.threshold).to_csv(file_name="dccm")


def run_lmi(args, backbone):
    residue_coordinate_matrix = ResidueCoordinateMatrix(args.ref, args.traj, args.atoms, backbone)
    LMIAnalyzer(residue_coordinate_matrix, threshold=args.threshold).to_csv(file_name="lmi")


def main():
    parser = argparse.ArgumentParser(description="Script to run correlation analysis using DCCM or LMI")
    subparsers = parser.add_subparsers()

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--atoms', type=str, required=False)
    common_args.add_argument('--backbone', action='store_true', required=False)
    common_args.add_argument("--ref", help="Reference file",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(
                            (".pdb, .gro, .psf, .top, .crd"), file_name).validate_file_extension(),
                        required=True)
    common_args.add_argument("--traj", help="a trajectory file",
                        type=lambda file_name: ArgumentParserFileExtensionValidation(
                            (".trj, .pdb, .xtc, .dcd"), file_name).validate_file_extension(),
                        required=True)
    common_args.add_argument("--threshold", type=float, default=0, help="Threshold for the correlation analysis")

    dccm_parser = subparsers.add_parser("dccm", help="Run DCCM correlation analysis", parents=[common_args])
    dccm_parser.set_defaults(func=run_dccm)

    lmi_parser = subparsers.add_parser("lmi", help="Run LMI correlation analysis", parents=[common_args])
    lmi_parser.set_defaults(func=run_lmi)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return

    backbone = False
    if (args.atoms and args.backbone) or not args.atoms:
        logger.warning(f"Backbone atoms will be utilized to compute dccm.")
        backbone = True
    args.func(args, backbone)


if __name__ == "__main__":
    main()
