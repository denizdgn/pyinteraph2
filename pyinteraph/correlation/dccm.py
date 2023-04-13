import numpy as np

from pyinteraph.correlation.base_correlation_analyzer import BaseCorrelationAnalyzer


class DCCMAnalyzer(BaseCorrelationAnalyzer):
    def _compute_analyzer(self, residue_i_coords, residue_j_coords):
        return np.corrcoef(residue_i_coords, residue_j_coords)[1][0]
