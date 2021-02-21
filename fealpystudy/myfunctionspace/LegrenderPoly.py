import numpy as np
import sympy as sp


class Legrendrepoly:
    '''
        d维单纯型上的n次Legrendre多项式
        Legrendre polynomials of degree n on d-dimensional simple form
    '''
    def __init__(self,d=None,n=None):
        if d is None:
            d = 1
        
        if n is None:
            n = 1

        self.d = d
        self.n = n
        self.beta = self.multindex(d=None,n=None)
        self.L_beta = self.Legrendrepolynomial(beta=None)

    def beta(self):
        return self.beta

    def L_beta(self):
        return self.L_beta

    def Legrendrepolynomial(self,beta=None):
        if beta is None:
            beta = self.beta  #(...,d)

        shape = np.shape(beta)
        d = shape[-1]
        shape = np.array(shape)
        shape[-1] = d+1
        L = np.shape(shape)[0]-1
        betas = np.zeros(shape,dtype=int)
        betas[...,1:] = beta
        beta = betas
        lam = np.array(sp.symbols('la:%d'%(d+1)))
       # Lam_beta = np.broadcast_to(np.expand_dims(lam,axis=np.arange(L)),shape)

       # print(Lam_beta**beta)
       # print(beta)
        vert_underline_beta = self.vert_underline_beta(beta)
        vert_overline_lambda = self.vert_overline_lambda(lam)

        derivatives_multindex = np.zeros(vert_underline_beta.shape,dtype=int)
        derivatives_multindex[...,0] = vert_underline_beta[...,0]
        for i in range(1,d+1):
            derivatives_multindex[...,i] = vert_underline_beta[...,i]+vert_underline_beta[...,i-1]+i-1

        Lam_beta = np.broadcast_to(np.expand_dims(lam,axis=np.arange(L)),shape)**derivatives_multindex\
                        *np.broadcast_to(np.expand_dims(1-vert_overline_lambda,axis=np.arange(L)),shape)**beta #shape=(...,d+1)

        lam = np.broadcast_to(np.expand_dims(lam,axis=np.arange(L)),shape)

        #print(Lam_beta[-1,...],derivatives_multindex[-1,...])
        lam = lam.reshape(-1)
        Lam_beta = Lam_beta.reshape(-1)
        derivatives_multindex = derivatives_multindex.reshape(-1)

        for i in range(len(lam)):
            Lam_beta[i] = sp.diff(Lam_beta[i],lam[i],derivatives_multindex[i])/np.math.factorial(derivatives_multindex[i])

        Lam_beta = Lam_beta.reshape(shape)
        Lam_beta = np.prod(Lam_beta,axis=-1)

        return Lam_beta


    def vert_underline_beta(self,beta):
        vert_underline_beta = np.zeros(beta.shape,dtype=int)
        vert_underline_beta[...,0] = beta[...,0]
        D = np.shape(vert_underline_beta)[-1]
        for i in range(1,D):
            vert_underline_beta[...,i] = beta[...,i]+vert_underline_beta[...,i-1]
        return vert_underline_beta


    def vert_overline_lambda(self,lam):
        vert_overline_lambda = np.zeros(lam.shape,dtype=object)
        vert_overline_lambda[...,-1] = lam[...,-1]
        D = np.shape(vert_overline_lambda)[-1]
        for i in range(D-2,-1,-1):
            vert_overline_lambda[...,i] = vert_overline_lambda[...,i+1]+lam[...,i]

        return vert_overline_lambda








    def multindex(self,d=None,n=None):
        '''homogeneous polynomials of degree n, multiple indicators '''
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
    Leg = Legrendrepoly(d=2,n=3)
    print(Leg.L_beta)
  #  print(Leg.beta)