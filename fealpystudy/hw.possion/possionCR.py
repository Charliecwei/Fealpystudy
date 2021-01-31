import sys
import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from pde_model1 import Laplace
from fealpy.mesh import MeshFactory
from fealpy.functionspace import CrouzeixRaviartFiniteElementSpace
import fealpy.boundarycondition as bdc

from fealpy.tools.show import showmultirate, show_error_table


#q = int(sys.argv[1])
n = int(sys.argv[1])

x, y = sp.symbols("x0,x1")
u = sp.cos(sp.pi*x)*sp.sin(sp.pi*y)


pde = Laplace(u,x,y,Dirichletbd = '(y == 1.0) | (y == 0.0) |(x == 1.0) | (x == 0.0)',)
domain = pde.domain()

mf = MeshFactory()
mesh = mf.boxmesh2d(domain, nx = n, ny = n, meshtype='tri')

NDof = np.zeros(4, dtype=mesh.itype)
errormatrix = np.zeros((2,4),dtype = mesh.ftype)
errortype = ['$||u-u_h||_0$', '$||\\nabla u-\\nabla u_h||_0$']


for i in range(4):
    print('Step:', i)
    space = CrouzeixRaviartFiniteElementSpace(mesh)

    NDof[i] = space.number_of_global_dofs()
    uh = space.function()
    A = space.stiff_matrix()



    qf = mesh.integrator(2, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()  #(NQ, TD+1)
    cellmeasure = mesh.entity_measure('cell') #(NC. )
    phi = space.basis(bcs)# (NQ. NC, ldof)
    M = np.einsum('i, ijk, ijm, j->jkm', ws, phi, phi, cellmeasure)

    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:,:, None],shape=M.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape=M.shape)

    M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))


    F = space.source_vector(pde.source)
   

    A = np.add(A,3*M)




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
        

