import numpy as np

from prg.models.linear.base_model_linear import LinearAmQ

__all__ = ["Model_x2_y2_AQ_pairwise"]


class Model_x2_y2_AQ_pairwise(LinearAmQ):

    def __init__(self):
        dim_x, dim_y = 2, 2
        A = np.array(
            [
                [
                    0.6323337679269881,
                    -0.09843546284224247,
                    -0.20273794002607562,
                    0.1434159061277705,
                ],
                [
                    -0.09843546284224246,
                    0.6323337679269883,
                    0.14341590612777047,
                    -0.20273794002607565,
                ],
                [
                    -0.028683181225554088,
                    -0.009452411994784882,
                    0.2040417209908735,
                    0.05019556714471971,
                ],
                [
                    -0.009452411994784863,
                    -0.028683181225554123,
                    0.05019556714471966,
                    0.20404172099087356,
                ],
            ]
        )
        mQ, mz0, Pz0 = LinearAmQ._init_random_params(dim_x, dim_y, val_max=1.0)
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            A=A,
            mQ=mQ,
            mz0=mz0,
            Pz0=Pz0,
            pairwiseModel=True,
        )
