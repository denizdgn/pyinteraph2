import argparse
from collections import OrderedDict
import functools

import numpy as np
import MDAnalysis as mda
import itertools


class DCCMCalculation:
    selected_atoms_mapping = {
        "CA": "name CA",
    }

    def __init__(self, ref, traj, atoms):
        self.trajectory = mda.Universe(ref, traj)
        self.n_frames = self.trajectory.trajectory.n_frames
        self.n_residues = self.trajectory.residues.residues.n_residues
        self.selected_atoms = self.selected_atoms_mapping.get(atoms, self.selected_atoms_mapping["CA"])

    @property
    @functools.lru_cache()
    def residues_with_atom_number(self):
        residues = OrderedDict()
        for res in self.trajectory.residues:
            residues[f"{res.resnum}{res.resname}"] = res.atoms.select_atoms(self.selected_atoms).atoms.ix_array
        return residues

    @property
    @functools.lru_cache()
    def coordinates_by_residue(self):
        traj_by_res = np.zeros(shape=(self.n_residues, self.n_frames,))
        for i, traj in enumerate(self.trajectory.trajectory):
            for res_num in range(0, self.n_residues):
                # np.mean to compute geometric center
                traj_by_res[res_num][traj.frame] = np.mean(traj.positions[list(self.residues_with_atom_number.values())[res_num]])
        return traj_by_res

    def compute(self):
        comb = list(itertools.combinations_with_replacement(list(range(0, self.n_residues)), 2))
        corr = np.zeros(shape=(self.n_residues, self.n_residues))
        for i, j in comb:
            corr[i, j] = np.corrcoef(self.coordinates_by_residue[i, :], self.coordinates_by_residue[j, :])[1][0]
        return corr


parser = argparse.ArgumentParser()
parser.add_argument('--ref', type=str, required=True)
parser.add_argument('--traj', type=str, required=True)
parser.add_argument('--atoms', type=str, required=False)
args = parser.parse_args()


dccm = DCCMCalculation(args.ref, args.traj, args.atoms)
corr = dccm.compute()
print(corr)
