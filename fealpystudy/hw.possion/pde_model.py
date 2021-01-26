import numpy as np
from fealpy.decorator import cartesian




class SinCosData:
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])


    @cartesian
    def solution(self, p):
        """ The exact solution """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        Val = np.sin(pi*x)*np.cos(pi*y)
        return Val

    @cartesian
    def source(self, p):
        """ The right hand side """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (2*pi*pi+3)*np.sin(pi*x)*np.cos(pi*y)
        return val 

    @cartesian
    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] =  pi*np.cos(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.sin(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        y = p[..., 1]
        return (y == 1.0) | (y == 0.0)

    @cartesian
    def neumann(self, p, n):
        """ Neumann bounadry condition """
        grad = self.gradient(p) #(NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) #(NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return x == 1.0

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
        x = p[..., 0]
        return x == 0.0
