# from linear import ModelFactory
# import numpy as np

from linear import BaseModelLinear, all_models


# Instancier un modèle directement depuis BaseModelLinear
m1 = BaseModelLinear(dim_x=2, dim_y=1, A=[[1,0],[0,1]], mQ=[[1,0],[0,1]], z00=[0,0], Pz00=[[1,0],[0,1]])
m1.info()

# Lister tous les fichiers de modèles détectés
print("Modèles détectés :", list(all_models.keys()))

# Importer un modèle spécifique et accéder à ses fonctions/classes
# Available : ['A_mQ_x1_y1', 'A_mQ_x3_y1', 'Sigma_x1_y1', 'Sigma_x3_y1', 'A_mQ_x2_y2', 'Sigma_x2_y2', 'A_mQ_x1_y1_VPgreaterThan1']
model_module = all_models['A_mQ_x1_y1']
# Par exemple si model1.py contient une fonction `create_model` :
my_model = model_module.create_model()
print(f'my_model={my_model.info}')
print(f'my_model={my_model.get_params()}')
