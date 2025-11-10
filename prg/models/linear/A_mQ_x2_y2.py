import numpy as np
from .base_model_linear import BaseModelLinear

# A few utils functions that are used several times
from others.Utils import check_consistency

class Model_A_mQ_x2_y2(BaseModelLinear):
    
    # Nom du modèle
    MODEL_NAME = "A_mQ_x2_y2"

    def __init__(self) -> None:
        super().__init__(dim_x=2, dim_y=2, model_type="AmQ")
    
        self.A = np.array( [
            [0.6323337679269881, -0.09843546284224247, -0.20273794002607562, 0.1434159061277705], 
            [-0.09843546284224246, 0.6323337679269883, 0.14341590612777047, -0.20273794002607565], 
            [-0.028683181225554088, -0.009452411994784882, 0.2040417209908735, 0.05019556714471971], 
            [-0.009452411994784863, -0.028683181225554123, 0.05019556714471966, 0.20404172099087356]] )

        self.mQ = np.array( [
            [0.7225554106910039, 0.3244784876140809, 0.5678943937418514, 0.1698174706649283],
            [0.3244784876140808, 0.7225554106910039, 0.1698174706649283, 0.5678943937418514],
            [0.5678943937418514, 0.1698174706649283, 0.957513037809648, 0.2719361147327249],
            [0.1698174706649283, 0.5678943937418514, 0.2719361147327249, 0.957513037809648]] )
        
        self.z00 = np.zeros(shape=(self.dim_x+self.dim_y, 1))
        self.Pz00 = np.eye(self.dim_x+self.dim_y)

        if __debug__:
            check_consistency(mQ=self.mQ, Pz00=self.Pz00)
