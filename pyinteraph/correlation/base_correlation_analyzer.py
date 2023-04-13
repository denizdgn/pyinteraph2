import abc

import numpy as np
import itertools

from pyinteraph.core import logger
from pyinteraph.correlation.residue_coordinate_matrix import ResidueCoordinateMatrix


class BaseCorrelationAnalyzer(abc.ABC):

    def __init__(self, residue_coordinate_matrix: ResidueCoordinateMatrix, threshold: float = 0) -> None:
        self.threshold = threshold
        self.residue_coordinate_matrix = residue_coordinate_matrix
        self.n_residues = self.residue_coordinate_matrix.n_residues
        self.coordinates_by_residue = self.residue_coordinate_matrix.coordinates_by_residue
        self.residues_with_atom_number = self.residue_coordinate_matrix.residues_with_atom_number

    def run(self) -> np.ndarray:
        residue_number_combinations = itertools.combinations_with_replacement(range(0, self.n_residues), 2)
        correlation_matrix = np.zeros(shape=(self.n_residues, self.n_residues))
        for i, j in residue_number_combinations:
            correlation_matrix[i, j] = correlation_matrix[j, i] = self._compute_analyzer(
                self.coordinates_by_residue[i, :], self.coordinates_by_residue[j, :]
            )
        return correlation_matrix

    @abc.abstractmethod
    def _compute_analyzer(self, residue_i_coords: np.ndarray, residue_j_coords: np.ndarray) -> float:
        raise NotImplementedError()

    @property
    def filtered_correlation_matrix(self):
        correlation_matrix = self.run()

        if self.threshold == 0:
            return correlation_matrix
        return np.where(correlation_matrix > self.threshold, correlation_matrix, 0)

    def to_csv(self, file_name="correlation") -> None:
        np.savetxt(f"{file_name}.csv", self.filtered_correlation_matrix, comments='', delimiter=',',
                   header=','.join(self.residues_with_atom_number.keys()))
