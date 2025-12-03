from nonLinear import ModelFactoryNonLinear
import numpy as np

if __name__ == "__main__":
    
    # Automatically detected
    print("Available models :", ModelFactoryNonLinear.list_models())

    # Available non linear models:
    # ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1', 'x2_y1_rapport', 'x2_y1_withRetroactionsOfObservations']
    model = ModelFactoryNonLinear.create("x2_y1_rapport")
    print(f'model={model}')
    print(f'model.model_type={model.model_type}')

    params = model.get_params()
    print(f'params={params}')
    
    # short-cuts
    dim_x, dim_y, g, mQ = params['dim_x'], params['dim_y'], params['g'], params['mQ']

     # data for test
    z     = np.random.rand(dim_x+dim_y).reshape(-1,1)
    noise = np.zeros_like(z)
    dt     = 1.0

    print(f"--- Model loaded automatically : {model.MODEL_NAME}---") # {model.__class__.__name__}
    print("mQ =\n", mQ)
    print("Result g(z):\n", g(z, noise, dt))
