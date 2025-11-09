import os
import importlib
from .base_model import BaseModel

__all__ = ['BaseModel', 'all_models', 'model_names']

# Dictionnaires pour stocker les modèles et leurs noms
all_models = {}
model_names = []

current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename not in ['__init__.py', 'base_model.py']:
        module_name = f"{__name__}.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
            # print(f'module=', module)

            # Vérifier si le module définit MODEL_NAME
            model_name = getattr(module, 'MODEL_NAME', None)
            if model_name:
                all_models[model_name] = module
                model_names.append(model_name)
            
            # print(f'all_models=', all_models)
            # input('ettteret')
        except Exception as e:
            print(f"Impossible de charger {module_name}: {e}")

# Rendre accessibles
__all__.extend(['all_models', 'model_names'])
