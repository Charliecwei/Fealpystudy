import numpy as np
import sympy as sp
from pde_model2 import generalelliptic



x, y = sp.symbols("x0,x1")
u = sp.cos(sp.pi*x)*sp.cos(sp.pi*y)
#a = np.array([[10.0,-1.0], [-1.0,2.0]],dtype=np.float64)
a = np.array([[1.0,0.0], [0.0,1.0]],dtype=np.float64)
#a = sp.Matrix([[x+100, y], [x, y+100]])
#b = np.array([1.0,1.0],dtype=np.float64)
b = sp.Matrix([[sp.cos(x)],[sp.sin(y)]])
#c = 1+x**2+y**2
c = np.array([0.0],dtype=np.float64)

pde = generalelliptic(u,x,y,a=a,b=b,c=c,Dirichletbd='(x==1.0)|(x==0.0)|(y==0.0)|(y==1.0)')


pde.test()


