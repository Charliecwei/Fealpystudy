import numpy as np
from scipy.sparse import csr_matrix, spdiags, bmat

from fealpy.decorator import barycentric
from fealpy.quadrature import FEMeshIntegralAlg

from fealpy.functionspace.Function import Function




class CR_FortinDof():
    """
        Define the Forin element's dof
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.itype = mesh.itype
        self.ftype = mesh.ftype
        self.cell2dof = self.cell_to_dof()


    def cell_to_dof(self, index=np.s_[:]):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        
        lNE = mesh.ds.cell_to_edge().shape[-1]
        lNF = mesh.ds.cell_to_face().shape[-1]
        lNC = 1

        shape = (NC,lNE+lNF+lNC)
        cell2dof = np.zeros(shape,dtype=self.itype)

        cell2dof[:,0:lNE] = mesh.ds.cell_to_edge()
        cell2dof[:,lNE:lNE+lNF] = NE + mesh.ds.cell_to_face()
        cell2dof[:,-1] = NE +NF + np.arange(0,NC)
 
        return cell2dof[index]

    def interpolation_points(self):
        mesh = self.mesh
        TNE = mesh.entity_barycenter('edge').shape[0]
        TNF = mesh.entity_barycenter('face').shape[0]
        TNC = mesh.entity_barycenter('cell').shape[0]
        GD = mesh.top_dimension()
        shape = (TNE+TNF+TNC,GD)  
        ipoints = np.zeros(shape,self.ftype)
        
        ipoints[0:TNE,:] = mesh.entity_barycenter('edge') 
        ipoints[TNE:TNE+TNF,:] = mesh.entity_barycenter('face')
        ipoints[TNE+TNF:,:] = mesh.entity_barycenter('cell')  #(gdof,GD)

        return ipoints











###################################################################
if __name__ == '__main__':
    from fealpy.mesh import MeshFactory

    mf = MeshFactory()
    mesh = mf.boxmesh3d([0,1,0,1,0,1], nx=1,ny=1,nz=1,meshtype='tet')
    
    dof = CR_FortinDof(mesh)
    ipoints = dof.interpolation_points()
    cell2dof = dof.cell_to_dof()
    print(ipoints.dtype, cell2dof.dtype, mesh.ftype)




