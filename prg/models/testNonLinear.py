from nonLinear import ModelFactory
import numpy as np


# Lister tous les fichiers de modèles détectés
print("Modèles détectés :", ModelFactory.list_models())


# Choisir un modèle par nom
# Available : ['x1_y1_cubique', 'x1_y1_ext_saturant', 'x1_y1_gordon', 'x1_y1_sinus', 'x2_y1_withRetroactionsOfObservations', 'x2_y1']
model = ModelFactory.create("x2_y1")
print(f'model={model}')

dim_x, dim_y, g, mQ, z00, Pz00, alpha, beta, kappa = model.get_params()
z     = np.random.rand(dim_x+dim_y)
noise = np.zeros_like(z)
dt   = 1.0

print(f"--- Model loaded automatically : {model.__class__.__name__} ---")
print("mQ =\n", mQ)
print("Résultat g(z):\n", g(z, noise, dt))

