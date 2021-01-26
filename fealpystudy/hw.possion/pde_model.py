import sympy as sp
import numpy as np
from fealpy.decorator import cartesian



class SinCosData:
    """
    -\Delta u + 3u = f
    u = sin(pi*x)*cos(pi*y)
    """

    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])

    #def domains(self):
        #return np.array([0, 1, 0, 1])

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





class SinSinData:
    """
    -\Delta u + 3u = f
    u = sin(pi*x)*cos(pi*y)
    """

    def __init__(self):
        x,y = sp.symbols("x,y")
        pi = np.pi

        u = sp.lambdify((x,y),sp.sin(pi*x)+sp.sin(pi*y),"numpy")
       # Du = sp.lambdify(np.array([[sp.diff(u,x,1)],[sp.diff(u,y,2)]]))
        #f = sp.lambdify(-(sp.diff(u,x,2)+sp.diff(u,y,2))+3*u)

        self.u = u
        #self.Du = Du
       # self.f = f




    def domain(self):
        return np.array([0, 1, 0, 1])

    #def domains(self):
        #return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ The exact solution """
        x = p[..., 0]
        y = p[..., 1]
        Val = self.u(x,y)
        return Val

    @cartesian
    def source(self, p):
        """ The right hand side """
        x = p[..., 0]
        y = p[..., 1]
        val = self.f(x,y)
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




class GeExpCosData:
    """
    -\grad (a*\grad u) + b*\grad u + c*u = f
    u = exp(x)*cos(pi*y)
    """

    def __init__(self, a=np.array([[1.0, 0.0], [0.0, 1.0]], dtype = np.float64), 
    b = np.array([[0],[0]], dtype = np.float64), c = np.array([0],dtype = np.float64)):
        self.a = a
        self.b = b
        self.c = c

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """the exact solution"""
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.exp(pi*x)*np.cos(pi*y)
        return val



