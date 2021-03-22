import numpy as numpy
from scipy.fftpack import fft,ifft
import sympy as sp
import matplotlib.pyplot as plt



class Laplacemode():
    """初始化方程模型"""
    def __init__(self,u,x,L=None):
        if L is None:
            L = 2*sp.pi
        
        self.L = L
        self.f = sp.lambdify(x,-sp.diff(u,x,2),'numpy')
        self.u = sp.lambdify(x,u-sp.integrate(u,(x,0,L))/L,'numpy')
        self.x = x
        self.spu = u-sp.integrate(u,(x,0,L))/L

    def source(self,p):
        return self.f(p)

    def solution(self,p):
        return self.u(p)

    def domian(self):
        return self.L

    def Fourierapprox(self,u=None,x=None,L=None,K=None):
        if u is None:
            u = self.spu
        if L is None:
            L = self.L
        if K is None:
            K = 10
        if x is None:
            x = self.x
        Fur = np.zeros(2*K+1,dtype=complex)
        for k in range(-K,K+1):
            Fur[k+K] = sp.integrate(u*sp.exp(k*x*1j),(x,0,L))/L
            
        return Fur


class Fourier_wave_function():
    """生成Fourier波形函数"""
    def __init__(self,u):
       N = u.shape[-1]
       self.u = u 
       self.N = N
       self.k = np.arange(self.N)
       

    def solution(self,x=None):
        if x is None:
            x = 0
        return np.sum(np.exp(x[...,None]*self.k*1j)*u,axis=-1)

    def source(self,x=None):
        if x is None:
            x = 0
        k = self.k**2
        f = u*k
        return np.sum(np.exp(x[...,None]*self.k*1j)*f,axis=-1)   



class Spectral_method_Laplace():
    """一维度周期边界"""
    def __init__(self,f,uo=None,L=None,N=None):
        if L is None:
            L = 2*np.pi
        if N is None:
            N = 4
        x = np.arange(N)*L/N
        disert_f = f(x)
        
        if disert_f is np.ndarray:
            tilde_f = fft(disert_f,n=N)
        else:
            tilde_f = fft(disert_f+0*x,n=N)
        k = np.arange(N)
        k[N//2+1:] = k[N//2+1:]-N
        #k = (L**2)/(4*np.pi**2)*(1/k**2)
        #tilde_u = tilde_f*k
        tilde_u = np.divide(tilde_f*((0.5*L/np.pi)**2),k**2)

        if uo is None:
            tilde_u[0] = 0
        else:
            tilde_u[0] = uo - np.sum(tilde_u)


        self.u = ifft(tilde_u,n=N) 
        self.tilde_u = tilde_u
        self.L = L
        self.N = N
        self.x = x

    def Numerical_solution(self,x=None):
        if x is None:
            x = self.x
        return self.interpolation(tilde_u=self.tilde_u,x=x)

    def ux_j(self):
        return self.u


    def interpolation(self,tilde_u=None,x=None):
        L = self.L
        if tilde_u is None:
            tilde_u = self.tilde_u
        if x is None:
            x = self.x


        k = np.arange(N)
        exp = np.exp(x[...,None]*(2*np.pi/L)*k*1j)
        Iu = np.sum(exp*tilde_u,axis=-1)/N
        return Iu



    def Lperror(self,exactu,Iu=None,N=None,p=None):
        if N is None:
            N = self.N
        if p is None:
            p = 2
        if Iu is None:
            Iu = self.Numerical_solution

        L = self.L
        x = L*np.arange(N)/N
        return (L*np.mean(np.abs(Iu(x)-exactu(x))**p))**(1/p)

    








###################################################################################

if __name__ == '__main__':
    import numpy as np
    import sympy as sp


    x = sp.symbols('x')

    exactu = sp.sin(x)
    exactu = sp.exp(-(x-sp.pi)**2)
    #exactu = sp.cos(5*x)
    #exactu = (x-sp.pi)**2


    K = 100
    u = np.random.random(K)/(np.arange(K))**8
    u[0] = 0
    pde = Fourier_wave_function(u)
    pde = Laplacemode(exactu,x=x)
    S = 50
    err = np.zeros((2,S))
    for k in range(S):
        N = 2*(k+1)
        Spm = Spectral_method_Laplace(pde.source,N=N)
        X = 2*np.pi*np.arange(N)/N
        err[0,k] = (np.max(np.abs(Spm.ux_j()-pde.solution(X))))
        err[1,k] = (Spm.Lperror(pde.solution))


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.log10(err[0,:]),'b--',lw=1,label='max norm')
    ax.plot(np.log10(err[1,:]),lw=1,label='L2 norm')
    ax.legend(loc='best')
    plt.show()




    
   