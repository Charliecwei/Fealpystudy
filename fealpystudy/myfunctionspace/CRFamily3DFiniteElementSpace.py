import numpy as np
from scipy.sparse import csr_matrix, spdiags, bmat

from fealpy.decorator import barycentric
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.functionspace.Function import Function
from fealpy.functionspace.LagrangeFiniteElementSpace import LagrangeFiniteElementSpace




class CRFamily3DFiniteElementSpace():
    def __init__(self, mesh, p=3):
        'p=1，2，3，4'
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.space = LagrangeFiniteElementSpace(mesh,p=p)
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()
        self.p = p

        self.itype = mesh.itype
        self.ftype = mesh.ftype


    def cell_to_dof(self):
        mesh = self.mesh
        ldof = self.number_of_local_dofs()
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        c2d = self.space.cell_to_dof()
        dofFlags = self.dof_flags_1() # 把不同类型的自由度区分开来
        cell2dof = np.zeros((NC,ldof),dtype=int)

        idx, = np.nonzero(dofFlags[0]) # 局部顶点自由度的编号

        cell2dof[:,idx] = mesh.ds.cell_to_face() #变为面自由度编号



        base1 = NF #新的起始编号
        base0 = NN #原来起始编号
        p = self.p
        edof = p-1
        fdof = (p-1)*(p-2)//2
        tdof = (p-1)*(p-2)*(p-3)//6


        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        cell2dof[:,idx] = c2d[:,idx] - base0 + base1
        base0+=edof*NE
        base1+=edof*NE

        idx, = np.nonzero(dofFlags[2]) # 面内部自由度编号
        cell2dof[:,idx] = c2d[:,idx] - base0 + base1
        base0+=fdof*NF
        base1+=fdof*NF

        idx, = np.nonzero(dofFlags[3]) # 体内部自由度编号
        cell2dof[:,idx] = c2d[:,idx] - base0 + base1
        base0+=tdof*NC
        base1+=tdof*NC

        #新加体内部不连续自由度
        d_trivp = self.d_triv()
        if d_trivp > 0:
            cell2dof[:,-d_trivp:] = np.arange(NC*d_trivp,dtype='int').reshape(NC,d_trivp)+base1
            base1+=NC*d_trivp
        #print(base1,np.max(cell2dof))
        return cell2dof





 

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        NF = self.mesh.number_of_faces()
        NN = self.mesh.number_of_nodes()
        return self.space.number_of_global_dofs()-NN+NF+self.d_triv()*NC
    def number_of_local_dofs(self):
        return self.space.number_of_local_dofs()+self.d_triv()

    def d_triv(self):
        '新增加的体上的非协调bubble个数'
        p = self.p
        return p//2 - (p-1)//3
    def d_refl(self):
        '一个面上非协调bubble个数'
        p = self.p
        return (p+1)//3

    def dof_flags_1(self):
        """ 
        对标量空间中的自由度进行分类, 分为:
            点上的自由由度
            边内部的自由度
            面内部的自由度
            体内部的自由度

        Returns
        -------

        """
        gdim = self.geo_dimension() # the geometry space dimension
        dof = self.space.dof 
        isPointDof = dof.is_on_node_local_dof()
        isEdgeDof = dof.is_on_edge_local_dof()
        isEdgeDof[isPointDof] = False
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) > 0
        if gdim == 2:
            return isPointDof, isEdgeDof0, ~(isPointDof | isEdgeDof0)
        elif gdim == 3:
            isFaceDof = dof.is_on_face_local_dof()
            isFaceDof[isPointDof, :] = False
            isFaceDof[isEdgeDof0, :] = False

            isFaceDof0 = np.sum(isFaceDof, axis=-1) > 0
            return isPointDof, isEdgeDof0, isFaceDof0, ~(isPointDof | isEdgeDof0 | isFaceDof0)
        else:
            raise ValueError('`dim` should be 2 or 3!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def BT_ncval(self):
        p =self.p
        if p == 1:
            BT_ncval = np.array([[-2,1,1,1],       
                                 [1,-2,1,1],
                                 [1,1,-2,1],
                                 [1,1,1,-2]],dtype='float')
        elif p == 2:
            BT_ncval = np.array([[4/3,-1/2,-1/2,-1/2,-2/3,1,1,-2/3,1,-2/3],    
                                 [-2/3,-1/2,1,1,4/3,-1/2,-1/2,-2/3,1,-2/3],
                                 [-2/3,1,-1/2,1,-2/3,-1/2,1,4/3,-1/2,-2/3],
                                 [-2/3,1,1,-1/2,-2/3,1,-1/2,-2/3,-1/2,4/3]],dtype='float')
        elif p == 3:
            BT_ncval = np.array([[-10/3,5/27,5/27,5/27,-20/27,0,0,-20/27,0,-20/27,5/3,5/9,5/9,5/9,0,5/9,5/3,5/9,5/9,5/3],
                                 [5/3,-20/27,5/9,5/9,5/27,0,0,5/9,0,5/9,-10/3,5/27,5/27,-20/27,0,-20/27,5/3,5/9,5/9,5/3],
                                 [5/3,5/9,-20/27,5/9,5/9,0,0,5/27,0,5/9,5/3,-20/27,5/9,5/27,0,5/9,-10/3,5/27,-20/27,5/3],
                                 [5/3,5/9,5/9,-20/27,5/9,0,0,5/9,0,5/27,5/3,5/9,-20/27,5/9,0,5/27,5/3,-20/27,5/27,-10/3]],dtype='float')
        elif p == 4:
            BT_ncval = np.array([[8/3,-305/384,-305/384,-305/384,-5/24,65/192,65/192,-5/24,65/192,-5/24,-65/384,-65/384,-65/384,-65/384,0,-65/384,-65/384,-65/384,-65/384,-65/384,-4/3,185/192,185/192,5/12,0,5/12,185/192,0,0,185/192,-4/3,185/192,5/12,185/192,-4/3],
                                 [-4/3,-65/384,185/192,185/192,-5/24,-65/384,-65/384,5/12,0,5/12,-305/384,65/192,65/192,-65/384,0,-65/384,185/192,0,0,185/192,8/3,-305/384,-305/384,-5/24,65/192,-5/24,-65/384,-65/384,-65/384,-65/384,-4/3,185/192,5/12,185/192,-4/3],
                                 [-4/3,185/192,-65/384,185/192,5/12,-65/384,0,-5/24,-65/384,5/12,185/192,-65/384,0,65/192,0,0,-305/384,65/192,-65/384,185/192,-4/3,-65/384,185/192,-5/24,-65/384,5/12,-305/384,65/192,-65/384,185/192,8/3,-305/384,-5/24,-65/384,-4/3],
                                 [-4/3,185/192,185/192,-65/384,5/12,0,-65/384,5/12,-65/384,-5/24,185/192,0,-65/384,0,0,65/192,185/192,-65/384,65/192,-305/384,-4/3,185/192,-65/384,5/12,-65/384,-5/24,185/192,-65/384,65/192,-305/384,-4/3,-65/384,-5/24,-305/384,8/3]],dtype='float')
        return BT_ncval

    def BK_ncval(self):
        p = self.p
        if p == 1:
            BK_ncval = np.array([0,0,0,0],dtype='float')
        elif p == 2:
            BK_ncval = np.array([2,0,0,0,2,0,0,2,0,2],dtype='float')     
        elif p == 3:
            BK_ncval = np.array([2,-4/3,-4/3,-4/3,-4/3,8/9,8/9,-4/3,8/9,-4/3,2,-4/3,-4/3,-4/3,8/9,-4/3,2,-4/3,-4/3,2],dtype='float')
        elif p == 4:
            BK_ncval = np.array([2,1/32,1/32,1/32,1/2,1/32,1/32,1/2,1/32,1/2,1/32,1/32,1/32,1/32,0,1/32,1/32,1/32,1/32,1/32,2,1/32,1/32,1/2,1/32,1/2,1/32,1/32,1/32,1/32,2,1/32,1/2,1/32,2],dtype='float')
        d_trivp = self.d_triv()
        if d_trivp == 1:
            BK_ncval = BK_ncval[...,None]
        return BK_ncval

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
        phi0 = self.space.basis(bc) #(NQ,1,ldof)
        shape = list(phi0.shape)
        shape[-1]+=1
        #print(shape)
        phi = np.zeros(shape,dtype='float')
        dofFlags = self.dof_flags_1() # 把不同类型的自由度区分开来

        idx, = np.nonzero(dofFlags[0]) # 面不连续自由度的编号
        #构建面上不连续基函数
        
        #print(np.einsum('...j,ij->...i',phi0,BT_ncval).shape)
        BT_ncval = self.BT_ncval()
        phi[...,idx] = np.einsum('...j,ij->...i',phi0,BT_ncval)

        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        phi[...,idx] = phi0[...,idx]

        idx, = np.nonzero(dofFlags[2]) # 面内部自由度的编号
        phi[...,idx] = phi0[...,idx]

        d_trivp = self.d_triv()
        if d_trivp>0:
            BK_ncval = self.BK_ncval()
            phi[...,-d_trivp:] = np.einsum('...j,jl->...l',phi0,BK_ncval)
            
        return phi

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
        gphi0 = self.space.grad_basis(bc) #(NQ,NC,ldof)
        shape = list(gphi0.shape)
        shape[-2]+=1
        gphi = np.zeros(shape,dtype='float')
        dofFlags = self.dof_flags_1() # 把不同类型的自由度区分开来

        idx, = np.nonzero(dofFlags[0]) # 面不连续自由度的编号
        #构建面上不连续基函数
        
        #print(np.einsum('...j,ij->...i',phi0,BT_ncval).shape)
        BT_ncval = self.BT_ncval()
        gphi[...,idx,:] = np.einsum('...jk,ij->...ik',gphi0,BT_ncval)

        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        gphi[...,idx,:] = gphi0[...,idx,:]

        idx, = np.nonzero(dofFlags[2]) # 面内部自由度的编号
        gphi[...,idx,:] = gphi0[...,idx,:]

        d_trivp = self.d_triv()
        if d_trivp>0:
            BK_ncval = self.BK_ncval()
            gphi[...,-d_trivp:,:] = np.einsum('...jk,jl->...lk',gphi0,BK_ncval)

        return gphi

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc) #phi.shape = (1,ldof) or (NQ,1,ldof)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1 #uh.shape = (gdof,) or 
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof[index]])
       # print(uh[cell2dof], '\n', cell2dof)
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





    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array, coordtype='barycentric')
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)











###################################################################
if __name__ == '__main__':
    from fealpy.mesh import MeshFactory
    from fealpy.mesh.TetrahedronMesh import TetrahedronMesh
    import matplotlib.pyplot as plt
    import scipy as sp
    import scipy.sparse.linalg
    from VectorFiniteElementSpace import VectorFiniteElementSpace
    from scipy.sparse.linalg import spsolve
    from fealpy.functionspace import LagrangeFiniteElementSpace
    from CRFortin3DFiniteElementSpace import CRFortin3DFiniteElementSpace
    from fealpy.quadrature import FEMeshIntegralAlg

    


    def Inf_Sup(mesh,uspace,pspace,q=5):
        cellmeasure = mesh.entity_measure('cell')
        integralalg = FEMeshIntegralAlg(
                    mesh, q,
                    cellmeasure=cellmeasure)
        integrator = integralalg.integrator
        bcs, ws = integrator.get_quadrature_points_and_weights()


        uspace = VectorFiniteElementSpace(uspace)       
        duphi = uspace.div_basis(bcs) #(NQ,1,uldof)
        #uphi = uspace.basis(bcs)
        pphi= pspace.basis(bcs) #(NQ,1,pldof)
        #print(divubasis.shape,pbasis.shape,ubasis.shape)
        uldof = uspace.number_of_local_dofs()
        pldof = pspace.number_of_local_dofs()
        B = np.einsum('i,ijk,ijm,j->jkm',ws, pphi,duphi,cellmeasure)
        I = np.einsum('ij, k->ijk', pspace.cell_to_dof(), np.ones(uldof,dtype=int))
        J = np.einsum('ij, k->ikj', uspace.cell_to_dof(), np.ones(pldof,dtype=int))

        pgdof = pspace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        #print(ugdof,np.max(J))
        B = csr_matrix((B.flat, (I.flat, J.flat)), shape=(pgdof, ugdof))
        #print(B.shape)
        U, D, V = scipy.sparse.linalg.svds(B,k=pgdof-1)
        #print(D.shape,B.shape,pgdof)

        return np.min(D)



    mf = MeshFactory()

    mesh = mf.one_tetrahedron_mesh()

    #mesh.uniform_refine(1)



    








    N = 4
    beta = np.zeros((N,3))
    
    for i in range(N):

        uspace = CRFamily3DFiniteElementSpace(mesh,p=2)
        pspace = LagrangeFiniteElementSpace(mesh,p=1,spacetype='D')
        beta[i,0] = Inf_Sup(mesh,uspace,pspace)



        uspace = CRFamily3DFiniteElementSpace(mesh,p=3)
        pspace = LagrangeFiniteElementSpace(mesh,p=2,spacetype='D')
        beta[i,1] = Inf_Sup(mesh,uspace,pspace)


        uspace = CRFamily3DFiniteElementSpace(mesh,p=4)
        pspace = LagrangeFiniteElementSpace(mesh,p=3,spacetype='D',q=7)
        beta[i,2] = Inf_Sup(mesh,uspace,pspace)

        if i < N-1:
            mesh.uniform_refine()
    print(beta)


    
   


