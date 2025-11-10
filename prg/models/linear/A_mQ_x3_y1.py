import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_A_mQ_x3_y1(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x3_y1"

    def __init__(self) -> None:
        super().__init__(dim_x=3, dim_y=1, model_type="AmQ")
    
        self.A = np.array( [
            [0.6399176954732511, -0.1502057613168724, 0.07818930041152264, -0.18518518518518517],
            [0.26851851851851855, 0.5462962962962961, -0.09259259259259262, -0.08333333333333329],
            [0.22119341563786007, 0.17798353909465012, 0.36625514403292186, -0.06481481481481483],
            [-0.2777777777777778, 0.055555555555555546, -0.11111111111111113, 0.49999999999999994]] )
        self.mQ = np.array( [
            [0.7164609053497941, 0.24629629629629626, 0.21131687242798353, 0.6555555555555556],
            [0.24629629629629632, 0.5958333333333334, 0.14120370370370378, 0.225],
            [0.21131687242798355, 0.14120370370370378, 0.6734053497942387, 0.41944444444444445],
            [0.6555555555555556, 0.22500000000000003, 0.4194444444444445, 0.8500000000000001]] )
        
        self.z00 = np.zeros(shape=(self.dim_x+self.dim_y, 1))
        self.Pz00 = np.eye(self.dim_x+self.dim_y)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)
