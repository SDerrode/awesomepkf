from linear import ModelFactoryLinear
import numpy as np

if __name__ == "__main__":
    
    # Lister tous les fichiers de modèles détectés
    print("Modèles détectés :", ModelFactoryLinear.list_models())

    # Available : ['A_mQ_x1_y1', 'A_mQ_x1_y1_VPgreaterThan1', 'A_mQ_x2_y2', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x2_y2', 'Sigma_x3_y1']
    model = ModelFactoryLinear.create("A_mQ_x1_y1")
    print(f'model={model}')
    print(f'model.model_type={model.model_type}')
    

    params = model.get_params()
    print(f'params={params}')
    
    exit(1)
    
    dim_x, dim_y, g, mQ = params['dim_x'], params['dim_y'], params['g'], params['mQ']

    # test sur les données 
    z     = np.random.rand(dim_x+dim_y).reshape(-1,1)
    noise = np.zeros_like(z)
    dt     = 1.0

    print(f"--- Model loaded automatically : {model.MODEL_NAME}---") # {model.__class__.__name__}
    print("mQ =\n", mQ)
    print("Résultat g(z):\n", g(z, noise, dt))
    
    # # Instancier un modèle directement depuis BaseModelLinear
    # m1 = BaseModelLinear(dim_x=2, dim_y=1, A=[[1,0],[0,1]], mQ=[[1,0],[0,1]], z00=[0,0], Pz00=[[1,0],[0,1]])

    # # Lister tous les fichiers de modèles détectés
    # print("Modèles détectés :", list(all_models.keys()))

    # # Importer un modèle spécifique et accéder à ses fonctions/classes
    # # Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
    # model_module = all_models['A_mQ_x1_y1']
    # # Par exemple si model1.py contient une fonction `create_model` :
    # my_model = model_module.create_model()
    # print(f'my_model={my_model.info}')
    # print(f'my_model={my_model.get_params()}')
