import numpy as np

class BaseModel:
    def __init__(self, **kwargs):
        self.dim_x  = kwargs.get('dim_x')
        self.dim_y  = kwargs.get('dim_y')
        self.dim_xy = self.dim_x + self.dim_y

        if all(param in kwargs for param in ['A', 'mQ', 'z00', 'Pz00']):
            self.model_type = 'type1'
            self.A    = np.array(kwargs['A'])
            self.mQ   = np.array(kwargs['mQ'])
            self.z00  = np.array(kwargs['z00'])
            self.Pz00 = np.array(kwargs['Pz00'])
        elif all(param in kwargs for param in ['sxx', 'syy', 'a', 'b', 'c', 'd', 'e']):
            self.model_type = 'type2'
            self.sxx  = np.array(kwargs['sxx'])
            self.syy  = np.array(kwargs['syy'])
            self.a    = np.array(kwargs['a'])
            self.b    = np.array(kwargs['b'])
            self.c    = np.array(kwargs['c'])
            self.d    = np.array(kwargs['d'])
            self.e    = np.array(kwargs['e'])
        else:
            raise ValueError("Paramètres insuffisants ou incompatibles pour initialiser le modèle.")

    def info(self):
        print(f"Type de modèle : {self.model_type}")
        print(f"Dimensions : dim_x={self.dim_x}, dim_y={self.dim_y}")
        if self.model_type == 'type1':
            print("Paramètres : A, mQ, z00, Pz00")
        else:
            print("Paramètres : sxx, syy, a, b, c, d, e")

    def get_params(self):
        if self.model_type == 'type1':
            return {'dim_x':self.dim_x, 'dim_y':self.dim_y, 'A':self.A, \
                        'mQ':self.mQ, 'z00':self.z00, 'Pz00':self.Pz00}
        else:
            return {'dim_x':self.dim_x, 'dim_y':self.dim_y, 'sxx':self.sxx, \
                        'syy':self.syy, 'a':self.a, 'b':self.b, 'c':self.c, 'd':self.d, 'e':self.e}
