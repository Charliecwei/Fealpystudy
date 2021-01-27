import sympy as sp
import numpy as np
from fealpy.decorator import cartesian



class generalelliptic:
    def __init__(self, u, x, y, a = None, b = None, c = None,
    Dirichletbd = None, Neumannbd = None, Robinbd = None):
        Du0 = sp.diff(u,x,1)
        Du1 = sp.diff(u,y,1)
        #Du = sp.Matrix([Du0,Du1])
        self.dir = Dirichletbd
        self.neu = Neumannbd
        self.rob = Robinbd

        if a is None:
            a = np.array([[1.0,0.0],[0.0,1.0]],dtype=np.float64)


        if b is None:
            b = np.array([0.0,0.0],dtype=np.float64)

        if c is None:
            c = np.array([0.0],dtype=np.float64)

        
        f = -(sp.diff(a[0,0]*Du0+a[0,1]*Du1,x,1)+sp.diff(a[1,0]*Du0+a[1,1]*Du1,y,1))
        +b[0]*Du0+b[1]*Du1+c*u

        self.u = sp.lambdify((x,y), u,'numpy')
        self.f = sp.lambdify((x,y), f,'numpy')
        self.Du0 = sp.lambdify((x,y), Du0,'numpy')
        self.Du1 = sp.lambdify((x,y), Du1,'numpy')

        if type(a) is np.ndarray:
            self.a = a
        else:
            self.a = sp.lambdify((x,y), a,'numpy')

        if type(b) is np.ndarray:
            self.b = b
        else:
            self.b = sp.lambdify((x,y), b,'numpy')


        if type(c) is np.ndarray:
            self.c = c
        else:
            self.c = sp.lambdify((x,y), c,'numpy')





    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        return self.u(p[...,0],p[...,1])

    @cartesian
    def source(self, p):
        return self.f(p[...,0],p[...,1])

    @cartesian
    def gradient(self, p):
        val = np.zeros(p.shape, dtype=np.float64)
        val[...,0] = self.Du0(p[...,0],p[...,1])
        val[...,1] = self.Du1(p[...,0],p[...,1])
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        if self.dir is None:
            shape = p.shape[:-1]
            return np.zeros(shape,dtype = np.bool)
        else:
            x = p[..., 0]
            y = p[..., 1]
            return eval(self.dir)

    @cartesian
    def neumann(self, p, n):
        """ Neumann bounadry condition """
        grad = self.gradient(p) #(NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) #(NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        if self.neu is None:
            shape = p.shape[:-1]
            return np.zeros(shape,dtype = np.bool)
        else:
            x = p[..., 0]
            y = p[..., 0]
            return eval(self.neu)

    @cartesian
    def robin(self,p,n):
        grad = self.gradient(p)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([3.0], dtype=np.float64).reshape(shape)
        val += kappa*self.solution(p)
        return val, kappa

    @cartesian
    def is_robin_boundary(self, p):
        if self.rob is None:
            shape = p.shape[:-1]
            return np.zeros(shape,dtype = np.bool)
        else:
            x = p[..., 0]
            y = p[..., 1]
            return eval(self.rob)







    def test(self):
        print(self.a)