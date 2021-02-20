import numpy as np
import sympy as sp


class Legrendrepoly:
    def __init__(self,d=None,n=None):
        if d is None:
            d = 1
        
        if n is None:
            n = 1

        self.d = d
        self.n = n
        self.beta = self.multindex(d=None,n=None)
        self.Legrendrepolynomial(beta=None)

    def beta(self):
        return self.beta

    def Legrendrepolynomial(self,beta=None):
        if beta is None:
            beta = self.beta  #(...,d)

        shape = np.shape(beta)
        d = shape[-1]
        shape = shape[:-1]
       # print(shape)
        lam = sp.symbols('la:%d'%d)
        

        return lam
        







    def multindex(self,d=None,n=None):
        if n is None:
            n = self.n
        
        if d is None:
            d = self.d

        if d == 1:
            return np.array([n],dtype=int)
        else:
            totalnumber = self.Combination_number(d=d,n=n)
            multindex = np.zeros((totalnumber,d),dtype=int)
            globali = 0

            for i in range(n+1):
                localnumber = self.Combination_number(d=d-1,n=i)
                multindex[globali:globali+localnumber,0] = multindex[globali:globali+localnumber,0]+n-i
                multindex[globali:globali+localnumber,1:] = self.multindex(d=d-1,n=i)
                globali = globali+localnumber
            return multindex



    def Combination_number(self,d=None,n=None):
        if n is None:
            n = self.n
            
        
        if d is None:
            d = self.d

        return np.math.factorial(d+n-1)//(np.math.factorial(d-1)*np.math.factorial(n))         




###################################################################
if __name__ == '__main__':
    import numpy as np
    Leg = Legrendrepoly(n=2,d=2)
    Cunmber = Leg.Combination_number
    mutidex = Leg.multindex
  #  print(Leg.beta)