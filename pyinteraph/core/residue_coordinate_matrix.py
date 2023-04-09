from collections import OrderedDict
import functools

import numpy as np
import MDAnalysis as mda


class ResidueCoordinateMatrix:
    def __init__(self, ref: str, traj: str, atoms: str, backbone: bool) -> None:
        self.ref = ref
        self.traj = traj
        self.selected_atoms = "backbone" if backbone else f"name {atoms.replace(',', ' ')}"

    @property
    @functools.lru_cache()
    def trajectory(self):
        return mda.Universe(self.ref, self.traj)

    @property
    @functools.lru_cache()
    def n_frames(self):
        return self.trajectory.trajectory.n_frames

    @property
    @functools.lru_cache()
    def n_residues(self):
        return self.trajectory.residues.residues.n_residues

    @property
    @functools.lru_cache()
    def residues_with_atom_number(self):
        residues = OrderedDict()
        for res in self.trajectory.residues:
            residues[f"{res.resnum}{res.resname}"] = res.atoms.select_atoms(self.selected_atoms).atoms.ix_array
        return residues

    @property
    @functools.lru_cache()
    def coordinates_by_residue(self) -> np.ndarray:
        traj_by_res = np.zeros(shape=(self.n_residues, self.n_frames,))
        for i, traj in enumerate(self.trajectory.trajectory):
            for res_num in range(0, self.n_residues):
                # np.mean to compute geometric center
                traj_by_res[res_num][traj.frame] = np.mean(
                    traj.positions[list(self.residues_with_atom_number.values())[res_num]])
        return traj_by_res
