from linear import ModelFactoryLinear
import numpy as np

if __name__ == "__main__":
    
    # Lister tous les fichiers de modèles détectés
    print("Modèles détectés :", ModelFactoryLinear.list_models())

    # Available : ['A_mQ_x1_y1', 'A_mQ_x1_y1_VPgreaterThan1', 'A_mQ_x2_y2', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x2_y2', 'Sigma_x3_y1']
    model = ModelFactoryLinear.create("A_mQ_x1_y1")
    print(f'model={model}')
    print(f'model.model_type={model.model_type}')
    
    params = model.get_params().copy()
    print(f'params={params}')
    
    dim_x, dim_y, g, mQ = params['dim_x'], params['dim_y'], params['g'], params['mQ']

    # test sur les données 
    z     = np.random.rand(dim_x+dim_y).reshape(-1,1)
    noise = np.zeros_like(z)
    dt    = 1.0

    print(f"--- Model loaded automatically : {model.MODEL_NAME}---") # {model.__class__.__name__}
    print("mQ =\n", mQ)
    print("Résultat g(z):\n", g(z, noise, dt))
