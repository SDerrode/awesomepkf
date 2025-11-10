from nonLinear import ModelFactoryNonLinear
import numpy as np

if __name__ == "__main__":
    
    # Lister tous les fichiers de modèles détectés
    print("Modèles détectés :", ModelFactoryNonLinear.list_models())

    # Available : ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
    model = ModelFactoryNonLinear.create("x1_y1_ext_saturant")
    print(f'model={model}')

    params = model.get_params()
    dim_x, dim_y, g, mQ = params['dim_x'], params['dim_y'], params['g'], params['mQ']

    # test sur les données 
    z     = np.random.rand(dim_x+dim_y).reshape(-1,1)
    noise = np.zeros_like(z)
    dt     = 1.0

    print(f"--- Model loaded automatically : {model.MODEL_NAME}---") # {model.__class__.__name__}
    print("mQ =\n", mQ)
    print("Résultat g(z):\n", g(z, noise, dt))
