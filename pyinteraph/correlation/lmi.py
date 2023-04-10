import numpy as np

from pyinteraph.core import logger
from pyinteraph.correlation.base_correlation_analyzer import BaseCorrelationAnalyzer


class LMIAnalyzer(BaseCorrelationAnalyzer):
    def _compute_analyzer(self, residue_i_coords, residue_j_coords):
        pearson_coefficient = np.corrcoef(residue_i_coords, residue_j_coords)[1][0]
        mutual_info = - 3/2 * np.log(1 - pearson_coefficient**2)
        linear_mutual_info = 1 - (np.exp(-2 * (mutual_info/3)))
        return linear_mutual_info
