import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve

from fealpy.functionspace.Function import Function
from CRFamily3DFiniteElementSpace import CRFamily3DFiniteElementSpace


class VectorCRFamily3DFiniteElementSpace():
    def __init__(self, mesh, p=3, spacetype='C'):
        self.scalarspace = CRFamily3DFiniteElementSpace(
                mesh, p=p)
        self.mesh = mesh
        self.p = p
        #self.dof = self.scalarspace.dof
        self.TD = self.scalarspace.TD
        self.GD = self.scalarspace.GD

    def __str__(self):
        return "Vector Lagrange finite element space!"

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def vector_dim(self):
        return self.GD

    def cell_to_dof(self):
        GD = self.GD
        cell2dof = np.copy(self.scalarspace.cell_to_dof())
        cell2dof = cell2dof[..., np.newaxis]
        cell2dof = GD*cell2dof + np.arange(GD)
        NC = cell2dof.shape[0]
        return cell2dof.reshape(NC, -1)

    def number_of_global_dofs(self):
        return self.GD*self.scalarspace.number_of_global_dofs()

    def number_of_local_dofs(self):
        #print(self.GD)
        return self.GD*self.scalarspace.number_of_local_dofs()

    def basis(self, bcs):
        GD = self.GD
        phi = self.scalarspace.basis(bcs)
        shape = list(phi.shape[:-1])
        phi = np.einsum('...j, mn->...jmn', phi, np.eye(self.GD))
        shape += [-1, GD] 
        phi = phi.reshape(shape)
        #print(shape,phi.shape)
        return phi

    def div_basis(self, bcs, cellidx=None):
        if cellidx is None:
            gphi = self.scalarspace.grad_basis(bcs)
        else:
            gphi = self.scalarspace.grad_basis(bcs, cellidx=cellidx)
        shape = list(gphi.shape[:-2])
        shape += [-1]
        return gphi.reshape(shape)

    def function(self, dim=None):
        f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=self.mesh.ftype)
