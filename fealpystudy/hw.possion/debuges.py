import numpy as np
import sympy as sp
from pde_model2 import generalelliptic


p = np.array([[1,0],[0,1]],dtype=np.float64)  


x, y = sp.symbols("x0,x1")
a = sp.Matrix([[x*y,x],[y,x]])
u = sp.cos(sp.pi*x)*sp.sin(sp.pi*y)

pde = generalelliptic(u,x,y,a=a)
pde.test()
print(a)