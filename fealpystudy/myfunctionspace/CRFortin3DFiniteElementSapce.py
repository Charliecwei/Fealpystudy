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
        ''' 
        lNE = mesh.ds.cell_to_edge().shape[-1]
        lNF = mesh.ds.cell_to_face().shape[-1]
        lNC = 1

        shape = (NC,lNE+lNF+lNC)
        cell2dof = np.empty(shape,dtype=self.itype) #(NC, lNE+lNF+lNC)

        cell2dof[:,0:lNE] = mesh.ds.cell_to_edge()
        cell2dof[:,lNE:lNE+lNF] = NE + mesh.ds.cell_to_face()
        cell2dof[:,-1] = NE +NF + np.arange(0,NC)
        '''

        cell2dof = np.vstack((mesh.ds.cell_to_edge().T,NE+mesh.ds.cell_to_face().T,NE+NF+np.arange(0,NC))).T 
        # cell2dof.shape = (NC,ldof), ldof = LNE+LNF+LNC

        return cell2dof[index] 

    def interpolation_points(self):
        mesh = self.mesh
        '''
        TNE = mesh.entity_barycenter('edge').shape[0]
        TNF = mesh.entity_barycenter('face').shape[0]
        TNC = mesh.entity_barycenter('cell').shape[0]
        GD = mesh.top_dimension()
        shape = (TNE+TNF+TNC,GD)  
        ipoints = np.empty(shape,self.ftype)
        
        ipoints[0:TNE,:] = mesh.entity_barycenter('edge') 
        ipoints[TNE:TNE+TNF,:] = mesh.entity_barycenter('face')
        ipoints[TNE+TNF:,:] = mesh.entity_barycenter('cell')  #(gdof,GD)
        '''

        ipoints = np.vstack((mesh.entity_barycenter('edge'),mesh.entity_barycenter('face'),mesh.entity_barycenter('cell')))
        # ipoints.shape = (gof,GD)
        return ipoints

class CRFortin3DFiniteElementSpace():
    def __init__(self, mesh, q=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')

        self.dof = CR_FortinDof(mesh)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = 4
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

    def number_of_global_dofs(self):
        return self.mesh.number_of_edges()+self.mesh.number_of_faces()+self.mesh.number_of_cells()

    def number_of_local_dofs(self):
        return 11

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell_to_dof(index=index)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    
    @barycentric
    def basis(self, bc):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(1, ldof)` or `(NQ, 1, ldof)`
        """
        phie = 4*bc[...,[0,0,0,1,1,2]]*bc[...,[1,2,3,2,3,3]] #(NQ,6)
        phif = 2*(1-bc)**2-3*(bc[...,[1,2,3,0]]**2+bc[...,[2,3,0,1]]**2+bc[...,[3,0,1,2]]**2) #(NQ,4)
        phic = 2-4*np.sum(bc**2,-1) #(NQ,1)
        if len(bc.shape)==1:
            phi = np.concatenate([phie,phif,[phic]])
        else:
            phi = np.vstack((phie.T,phif.T,phic.T)).T

       # print(phi[..., None, :].shape) #(1, ldof) or (NQ, 1, ldof)
        return phi[..., None, :] 

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """
        compute the grad basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be (TD+1, ) or `(NQ, TD+1)`

        Returns
        -------
        gphi : numpy.ndarray
            the shape of `gphi` can be `(NC, ldof, GD)' or
            `(NQ, NC, ldof, GD)'
        """
        mesh = self.mesh
        grad_lambda = mesh.grad_lambda()[index] #(NC, TD+1, GD)
        gphie = 4*(bc[...,None,[0,0,0,1,1,2],None]*grad_lambda[...,[1,2,3,2,3,3],:]
                    +bc[...,None,[1,2,3,2,3,3],None]*grad_lambda[...,[0,0,0,1,1,2],:])
        gphif = 4*(bc[...,None,:,None]-1)*grad_lambda - 6*(bc[...,None,[1,2,3,0],None]*grad_lambda[...,[1,2,3,0],:]
                                + bc[...,None,[2,3,0,1],None]*grad_lambda[...,[2,3,0,1],:]
                                +bc[...,None,[3,0,1,2],None]*grad_lambda[...,[3,0,1,2],:])
        gphic = -8*np.sum(bc[...,None,:,None]*grad_lambda,-2)
        gphic = gphic[...,None,:]

        
        NQ = bc.shape[0]
        NC = grad_lambda.shape[0]
        ldof = self.number_of_local_dofs()
        GD = self.geo_dimension()
        if len(bc.shape) == 1:
            shape = (NC, ldof, GD)
        else:
            shape = (NQ,NC,ldof,GD)
        
        gphi = np.empty(shape)

        gphi[...,0:6,:] = gphie
        gphi[...,6:10,:] = gphif
        gphi[...,10:11,:] = gphic 
       # print(gphi.shape) #(NC, ldof, GD) or (NQ, NC, ldof, GD)
        return gphi

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc) #phi.shape = (1,ldof) or (NQ,1,ldof)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1 #uh.shape = (gdof,) or 
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        gphi = self.grad_basis(bc, index=index) #(NC, ldof, GD) or (NQ, NC, ldof, GD)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

    def stiff_matrix(self, c=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.grad_basis, cell2dof, gdof)
        q = 2 #gphi is pisecise linear
        A = self.integralalg.serial_construct_matrix(b0, c=c, q=q)
        return A

    def source_vector(self, f, dim=None, q=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        if q is None:
            q = 3

        b = self.integralalg.serial_construct_vector(f, b0, q=q)
        return q









###################################################################
if __name__ == '__main__':
    from fealpy.mesh import MeshFactory
    import matplotlib.pyplot as plt
    import sympy as sp
    from mpl_toolkits.mplot3d import Axes3D
    from pde_model2 import generalelliptic3D

    #load mesh
    mf = MeshFactory()
    mesh = mf.boxmesh3d([0,1,0,1,0,1], nx=2,ny=2,nz=2,meshtype='tet')
    #mesh = mf.one_tetrahedron_mesh()

    #load space
    space = CRFortin3DFiniteElementSpace(mesh)

    #check the dof and interpolation point
    dof = CR_FortinDof(mesh)
    ipoints = dof.interpolation_points()
    cell2dof = dof.cell_to_dof()

    #check the phi and gphi
    qf = mesh.integrator(4, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    bcs = np.array([1.0,0.0,0.0,0.0],dtype=mesh.ftype)
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    grad_lambda = mesh.grad_lambda()
    '''
    np.set_printoptions(precision=3)

    print('bc = ', bcs.squeeze(), 
        '\n phi = ', phi.squeeze(),
        '\n gphi = ', gphi.squeeze(),
        '\n grad_lambda = \n', grad_lambda.squeeze())

    '''
    #print(cell2dof.shape, phi.shape, gphi.shape)

    #check the value and grad_value
    gdof = space.number_of_global_dofs()
    uh = np.random.random((gdof,2))
    uhs = space.value(uh,bcs)
    guhs = space.grad_value(uh,bcs)

   # print(bcs.shape,cell2dof.shape,uh.shape,uhs.shape,guhs.shape)
    
    ''' stiff matrix '''
    x, y, z = sp.symbols('x0,x1,x2')
    u = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)*sp.sin(sp.pi*z)

    pde = generalelliptic3D(u,x,y,z,
          Dirichletbd='(x==1.0)|(x==0.0)|(y==0.0)|(y==1.0)|(z==0)|(z==1)')

    A = space.stiff_matrix()
    F = space.source_vector(pde.source)

    print(F)
'''    
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    face = mesh.entity('face')
    edge = mesh.entity('edge')

    cell2face = mesh.ds.cell_to_face()
    cell2edge = mesh.ds.cell_to_edge()

    #print(cell2edge,'\n\n', edge)

    #print(cell2dof)

    fig = plt.figure()
    axes = Axes3D(fig)
    mesh.add_plot(axes)


    mesh.find_node(axes, showindex=True, color='b', fontsize=30)
    #mesh.find_edge(axes, showindex=True, color='g', fontsize=24)
    mesh.find_face(axes, showindex=True, fontsize=22)
    #mesh.find_cell(axes, showindex=True, fontsize=20)

    mesh.find_node(axes, showindex=True, color='b', fontsize=30)
    mesh.find_node(axes, node=ipoints, showindex=True, color='r', fontsize=24)
    #plt.show()

'''