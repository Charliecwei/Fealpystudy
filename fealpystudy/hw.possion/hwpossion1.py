import sys
import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from pde_model1 import Laplace
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
import fealpy.boundarycondition as bdc

from fealpy.tools.show import showmultirate, show_error_table


p = int(sys.argv[1])
n = int(sys.argv[2])

x, y = sp.symbols("x0,x1")
u = sp.cos(sp.pi*x)*sp.sin(sp.pi*y)


pde = Laplace(u,x,y,Dirichletbd = '(y == 1.0) | (y == 0.0)',
Neumannbd = 'x == 0.0',Robinbd = 'x == 1.0')
domain = pde.domain()

mf = MeshFactory()
mesh = mf.boxmesh2d(domain, nx = n, ny = n, meshtype='tri')

NDof = np.zeros(4, dtype=mesh.itype)
errormatrix = np.zeros((2,4),dtype = mesh.ftype)
errortype = ['$||u-u_h||_0$', '$||\\nabla u-\\nabla u_h||_0$']


for i in range(4):
    print('Step:', i)
    space = LagrangeFiniteElementSpace(mesh, p=p)

    NDof[i] = space.number_of_global_dofs()
    uh = space.function()
    A = space.stiff_matrix()
    M = space.mass_matrix()
    F = space.source_vector(pde.source)

    A = np.add(A,3*M)

    bc = bdc.RobinBC(space, pde.robin, threshold=pde.is_robin_boundary)
    A, F = bc.apply(A, F)

    bc = bdc.NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)
    F = bc.apply(F)

    bc = bdc.DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    A, F = bc.apply(A,F,uh)


    uh[:] = spsolve(A, F)


    errormatrix[0, i] = space.integralalg.L2_error(pde.solution, uh.value)
    errormatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < 3:
        mesh.uniform_refine()

k=0
showmultirate(plt, k, NDof, errormatrix, errortype, propsize=20)
show_error_table(NDof, errortype,  errormatrix)

plt.show()
        

