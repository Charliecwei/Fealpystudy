import numpy as np
from pde_models import Laplaces
#from pde_models import test
from pde_model import SinCosData

p = np.array([[1,0],[0,1]],dtype=np.float64)  

pdes = SinCosData()
pde = Laplaces()
#ts =  test()
#s = pde.solution(p)
s = pde.gradient(p)
#s = ts.solution(p)
#s = pdes.solution(p)
print(type(s))