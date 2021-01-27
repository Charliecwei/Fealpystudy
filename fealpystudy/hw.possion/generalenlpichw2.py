import sys
import numpy as np
import sympy as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pde_model2 import generalelliptic
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
import fealpy.boundarycondition as bdc
from fealpy.tools.show import showmultirate, show_error_table

p = int(sys.argv[1])
n = int(sys.argv[2])

x, y = sp.symbols("x0,x1")
u = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
#a = np.array([[10.0,-1.0], [-1.0,1.0]],dtype=np.float64)
a = np.array([[0.0,0.0], [0.0,0.0]],dtype=np.float64)
#a = sp.Matrix([[x+100, y], [x, y+100]])
b = np.array([1.0,1.0],dtype=np.float64)
#b = sp.Matrix([sp.sin(x),sp.sin(y)])
#c = 1+x**2+y**2
c = np.array([0.0],dtype=np.float64)


pde = generalelliptic(u,x,y,a=a,b=b,c=c,Dirichletbd='(x==1.0)|(x==0.0)|(y==0.0)|(y==1.0)')
domain = pde.domain()

mf = MeshFactory()
mesh = mf.boxmesh2d(domain, nx = n, ny = n, meshtype='tri')

NDof = np.zeros(4,dtype = mesh.itype)
errormatrix = np.zeros((2,4),dtype = mesh.ftype)
errortype = ['$||u-u_h||_0$', '$||\\nabla u-\\nabla u_h||_0$']


for i in range(4):
    print('step', i)
    space =LagrangeFiniteElementSpace(mesh, p=p)

    NDof[i] = space.number_of_global_dofs()
    uh = space.function()

    #construct matrix A
    qf = mesh.integrator(2*p, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()  #(NQ, TD+1)
    cellmeasure = mesh.entity_measure('cell') #(NC. )
    gphi = space.grad_basis(bcs) #(NQ, NC, ldof, GD)
    ps = mesh.bc_to_point(bcs) # (NQ, NC, GD)

    As = pde.A(ps)

    if len(As.shape) == 2:
        A = np.einsum('i, ijkl, nl, ijmn, j->jkm', ws, gphi, As, gphi, cellmeasure) #(NC, ldof, ldof)
    else:
        A = np.einsum('i, ijkl, ijnl, ijmn, j->jkm', ws, gphi, As, gphi, cellmeasure)
    
    #construct matrix B
    phi = space.basis(bcs)# (NQ. NC, ldof)
    Bs = pde.B(ps) #(NQ, NC, GD)
    if len(Bs.shape) == 1:
        B = -np.einsum('i, ijk, l, ijml, j->jkm', ws, phi, Bs, gphi, cellmeasure)
    else:
        B = -np.einsum('i, ijk, ijl, ijml, j->jkm', ws, phi, Bs, gphi, cellmeasure)

    #construct matrxi divb
    Bs = pde.DivB(ps)  #(NQ, NC)
    if len(Bs.shape) == 1:
        DivB = -np.einsum('i, ijk, k, ijm, j->jkm', ws, phi, Bs, phi, cellmeasure)
    else:
        DivB = -np.einsum('i, ijk, ij, ijm, j->jkm', ws, phi, Bs, phi, cellmeasure)

    
    #construct matrix C
    Cs = pde.C(ps)  #(NQ, NC)

    if len(Cs.shape) == 1:
        C = np.einsum('i, ijk, k, ijm, j->jkm', ws, phi, Cs, phi, cellmeasure)
    else:
        C = np.einsum('i, ijk, ij, ijm, j->jkm', ws, phi, Cs, phi, cellmeasure)
    
    #print(C[1:5,...])
    #construct stiff matrix A
    A = A + B + DivB + C
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:,:, None],shape=A.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape=A.shape)

    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))

    #construct F
    val = pde.source(ps)

    bb = np.einsum('i, ij, ijm, j->jm', ws, val, phi, cellmeasure)
    F = np.zeros(gdof, dtype=np.float64)
    np.add.at(F, cell2dof, bb)
    
    #boundary
    bc = bdc.DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
    A, F = bc.apply(A,F,uh)

    

    uh[:] = spsolve(A, F)

   # print(np.dot(A*uh - F,A*uh - F))



    errormatrix[0, i] = space.integralalg.L2_error(pde.solution, uh.value)
    errormatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value)

    if i < 3:
        mesh.uniform_refine()



k=0
#showmultirate(plt, k, NDof, errormatrix, errortype, propsize=20)
show_error_table(NDof, errortype,  errormatrix)

#plt.show()