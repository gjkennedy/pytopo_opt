import numpy as np
from scipy import sparse
from scipy import spatial
from scipy.sparse import linalg
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.tri as tri
from mpi4py import MPI
from paropt import ParOpt

class NodeFilter:
    """
    A node-based filter for topology optimization
    """
    def __init__(self, conn, X, r0=1.0, ftype='spatial',
                 beta=10.0, eta=0.5, projection=False):
        """
        Create a filter
        """
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1

        # Store information about the projection
        self.beta = beta
        self.eta = eta
        self.projection = projection

        # Store information about the filter
        self.F = None
        self.A = None
        self.B = None
        if ftype == 'spatial':
            self._initialize_spatial(r0)
        else:
            self._initialize_helmholtz(r0)

        return

    def _initialize_spatial(self, r0):
        """
        Initialize the spatial filter
        """

        # Create a KD tree
        tree = spatial.KDTree(self.X)

        F = sparse.lil_matrix((self.nnodes, self.nnodes))
        for i in range(self.nnodes):
            indices = tree.query_ball_point(self.X[i, :], r0)
            Fhat = np.zeros(len(indices))

            for j, index in enumerate(indices):
                dist = np.sqrt(np.dot(self.X[i, :] - self.X[index, :],
                                      self.X[i, :] - self.X[index, :]))
                Fhat[j] = r0 - dist

            Fhat = Fhat/np.sum(Fhat)
            F[i, indices] = Fhat

        self.F = F.tocsr()
        self.FT = self.F.transpose()

        return

    def _initialize_helmholtz(self, r0):
        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.conn[index, :]:
                for jj in self.conn[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        i_index = np.array(i, dtype=int)
        j_index = np.array(j, dtype=int)

        # Quadrature points
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Assemble all of the the 4 x 4 element stiffness matrices
        Ae = np.zeros((self.nelems, 4, 4))
        Ce = np.zeros((self.nelems, 4, 4))

        Be = np.zeros((self.nelems, 2, 4))
        He = np.zeros((self.nelems, 1, 4))
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N = 0.25*np.array([(1.0 - xi)*(1.0 - eta),
                                   (1.0 + xi)*(1.0 - eta),
                                   (1.0 + xi)*(1.0 + eta),
                                   (1.0 - xi)*(1.0 + eta)])
                Nxi = np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
                Neta = np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1]/detJ
                invJ[:, 0, 1] = -J[:, 0, 1]/detJ
                invJ[:, 1, 0] = -J[:, 1, 0]/detJ
                invJ[:, 1, 1] = J[:, 0, 0]/detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                He[:, 0, :] = N
                Be[:, 0, :] = Nx
                Be[:, 1, :] = Ny

                Ce += np.einsum('n,nij,nil -> njl', detJ, He, He)
                Ae += np.einsum('n,nij,nil -> njl', detJ * r0**2, Be, Be)

        # Finish the computation of the Ae matrices
        Ae += Ce

        A = sparse.coo_matrix((Ae.flatten(), (i_index, j_index)))
        A = A.tocsc()
        self.A = linalg.factorized(A)

        B = sparse.coo_matrix((Ce.flatten(), (i_index, j_index)))
        self.B = B.tocsr()
        self.BT = self.B.transpose()

        return

    def apply(self, x):
        if self.F is not None:
            rho = self.F.dot(x)
        else:
            rho = self.A(self.B.dot(x))

        if self.projection:
            denom = np.tanh(self.beta*self.eta) + np.tanh(self.beta*(1.0 - self.eta))
            rho = (np.tanh(self.beta*self.eta) + np.tanh(self.beta*(rho - self.eta)))/denom

        return rho

    def applyGradient(self, g, rho=None):
        if self.projection:
            denom = np.tanh(self.beta*self.eta) + np.tanh(self.beta*(1.0 - self.eta))
            grad = (self.beta/denom) * 1.0/np.cosh(self.beta*(rho - self.eta))**2
        else:
            grad = g

        if self.F is not None:
            return self.FT.dot(grad)
        else:
            return self.BT.dot(self.A(grad))

    def plot(self, u, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2*self.nelems, 3), dtype=int)
        triangles[:self.nelems, 0] = self.conn[:, 0]
        triangles[:self.nelems, 1] = self.conn[:, 1]
        triangles[:self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems:, 0] = self.conn[:, 0]
        triangles[self.nelems:, 1] = self.conn[:, 2]
        triangles[self.nelems:, 2] = self.conn[:, 3]

        # Create the triangulation object
        tri_obj = tri.Triangulation(self.X[:,0], self.X[:,1], triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect('equal')

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)

        return

class TopologyOptimization(ParOpt.Problem):
    def __init__(self, fltr, conn, X, bcs, forces={},
                 E=10.0, nu=0.3, P=3.0, area_fraction=0.4,
                 draw_history=True):

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.P = P
        self.draw_history = draw_history

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2*self.nnodes

        # Set up the topology optimization problem
        super(TopologyOptimization, self).__init__(MPI.COMM_SELF, self.nnodes, 1)

        # Compute the constitutivve matrix
        self.C0 = E*np.array([[1.0, nu, 0.0],
                              [nu, 1.0, 0.0],
                              [0.0, 0.0, 0.5*(1.0 - nu)]])
        self.C0 *= 1.0/(1.0 - nu**2)

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.f = self._compute_forces(self.nvars, forces)

        self.area_gradient = self.eval_area_gradient()
        self.fixed_area = area_fraction*np.sum(self.area_gradient)
        self.area_gradient_rho = np.zeros(self.nnodes)

        for k in range(4):
            np.add.at(self.area_gradient_rho, self.conn[:, k], self.area_gradient)
        self.area_gradient_rho[:] *= 0.25

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
        self.var[:, ::2] = 2*self.conn
        self.var[:, 1::2] = 2*self.conn + 1

        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.var[index, :]:
                for jj in self.var[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)

    def _compute_reduced_variables(self, nvars, bcs):
        """
        Compute the reduced set of variables
        """
        reduced = list(range(nvars))

        # For each node that is in the boundary condition dictionary
        for node in bcs:
            uv_list = bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in uv_list:
                var = 2*node + index
                reduced.remove(var)

        return reduced

    def _compute_forces(self, nvars, forces):
        """
        Unpack the dictionary containing the forces
        """
        f = np.zeros(nvars)

        for node in forces:
            f[2*node] += forces[node][0]
            f[2*node+1] += forces[node][1]

        return f

    def assemble_stiffness_matrix(self, C):
        """
        Assemble the stiffness matrix
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8))
        Be = np.zeros((self.nelems, 3, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
                Neta = np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1]/detJ
                invJ[:, 0, 1] = -J[:, 0, 1]/detJ
                invJ[:, 1, 0] = -J[:, 1, 0]/detJ
                invJ[:, 1, 1] = J[:, 0, 0]/detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                # This is a fancy (and fast) way to compute the element matrices
                Ke += np.einsum('n,nij,nik,nkl -> njl', detJ, Be, C, Be)

                # This is a slower way to compute the element matrices
                # for k in range(self.nelems):
                #     Ke[k, :, :] += detJ[k]*np.dot(Be[k, :, :].T, np.dot(self.C[k, :, :], Be[k, :, :]))

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def adjoint_residual_product(self, psi, u):
        dfdC = np.zeros((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        psie = np.zeros((self.nelems, 8))

        ue[:, ::2] = u[2*self.conn]
        ue[:, 1::2] = u[2*self.conn+1]

        psie[:, ::2] = psi[2*self.conn]
        psie[:, 1::2] = psi[2*self.conn+1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
                Neta = np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1]/detJ
                invJ[:, 0, 1] = -J[:, 0, 1]/detJ
                invJ[:, 1, 0] = -J[:, 1, 0]/detJ
                invJ[:, 1, 1] = J[:, 0, 0]/detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                for k in range(self.nelems):
                    eu = np.dot(Be[k, :], ue[k, :])
                    epsi = np.dot(Be[k, :], psie[k, :])

                    dfdC[k, :, :] += detJ[k]*np.outer(epsi, eu)

        return dfdC

    def reduce_vector(self, forces):
        """
        Eliminate essential boundary conditions from the vector
        """
        return forces[self.reduced]

    def reduce_matrix(self, matrix):
        """
        Eliminate essential boundary conditions from the matrix
        """
        temp = matrix[self.reduced, :]
        return temp[:, self.reduced]

    def solve(self, C):
        """
        Perform a linear static analysis
        """

        K = self.assemble_stiffness_matrix(C)
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.f)

        ur = sparse.linalg.spsolve(Kr, fr)

        u = np.zeros(self.nvars)
        u[self.reduced] = ur

        return u

    def compliance(self, x):
        # Compute the density at each node
        self.rho = self.fltr.apply(x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25*(self.rho[self.conn[:, 0]] + self.rho[self.conn[:, 1]] +
                          self.rho[self.conn[:, 2]] + self.rho[self.conn[:, 3]])

        # Compute the element stiffnesses
        self.C = np.outer(self.rhoE**self.P, self.C0)
        self.C = self.C.reshape((self.nelems, 3, 3))

        self.u = self.solve(self.C)

        return self.f.dot(self.u)

    def compliance_gradient(self):
        # Compute the gradient of the compliance
        dfdC = -1.0*self.adjoint_residual_product(self.u, self.u)

        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i,j]*dfdC[:,i,j]

        dfdrhoE[:] *= self.P*(self.rhoE)**(self.P - 1.0)

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.applyGradient(dfdrho)

    def eval_area_gradient(self):

        dfdrhoE = np.zeros(self.nelems)

        # Quadrature points
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Jacobian transformation
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
                Neta = np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]

                dfdrhoE[:] += detJ

        return dfdrhoE

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = 1e-3
        ub[:] = 1.0
        x[:] = 0.95
        return

    def evalObjCon(self, x):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        obj = self.compliance(x[:])
        con = [self.fixed_area - self.area_gradient.dot(self.rhoE)]

        if self.draw_history:
            fig, ax = plt.subplots()
            self.plot(self.rho, ax=ax)
            ax.set_aspect('equal', 'box')
            plt.savefig('topology.png')
            plt.close()

        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        g[:] = self.compliance_gradient()
        A[0][:] = -self.fltr.applyGradient(self.area_gradient_rho[:])

        return fail

    def plot(self, u, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2*self.nelems, 3), dtype=int)
        triangles[:self.nelems, 0] = self.conn[:, 0]
        triangles[:self.nelems, 1] = self.conn[:, 1]
        triangles[:self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems:, 0] = self.conn[:, 0]
        triangles[self.nelems:, 1] = self.conn[:, 2]
        triangles[self.nelems:, 2] = self.conn[:, 3]

        # Create the triangulation object
        tri_obj = tri.Triangulation(self.X[:,0], self.X[:,1], triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect('equal')

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)

        return

lx = 10
m = 96

ly = 10
n = 96

nelems = m*n
nnodes = (m + 1)*(n + 1)

y = np.linspace(0, ly, n + 1)
x = np.linspace(0, lx, m + 1)
nodes = np.arange(0, (n + 1)*(m + 1)).reshape((n + 1, m + 1))

# Set the node locations
X = np.zeros((nnodes, 2))
for j in range(n + 1):
    for i in range(m + 1):
        X[i + j*(m + 1), 0] = x[i]
        X[i + j*(m + 1), 1] = y[j]

# Set the connectivity
conn = np.zeros((nelems, 4), dtype=int)
for j in range(n):
    for i in range(m):
        conn[i + j*m, 0] = nodes[j, i]
        conn[i + j*m, 1] = nodes[j, i + 1]
        conn[i + j*m, 2] = nodes[j + 1, i + 1]
        conn[i + j*m, 3] = nodes[j + 1, i]

# Set the constrained degrees of freedom at each node
bcs = {}
for j in range(n):
    bcs[nodes[j, 0]] = [0, 1]

P = 10.0
forces = {}
pn = n//10
for j in range(pn):
    forces[nodes[j, -1]] = [0, -P/pn]

# Create the filter
r0 = 0.05*np.min((lx, ly))
fltr = NodeFilter(conn, X, r0, ftype='spatial')
topo = TopologyOptimization(fltr, conn, X, bcs, forces)

topo.checkGradients()

options = {
    'algorithm': 'tr',
    'tr_init_size': 0.05,
    'tr_min_size': 1e-6,
    'tr_max_size': 10.0,
    'tr_eta': 0.25,
    'tr_infeas_tol': 1e-6,
    'tr_l1_tol': 1e-3,
    'tr_linfty_tol': 0.0,
    'tr_adaptive_gamma_update': True,
    'tr_max_iterations': 1000,
    'max_major_iters': 100,
    'penalty_gamma': 1e3,
    'qn_subspace_size': 10,
    'qn_type': 'bfgs',
    'abs_res_tol': 1e-8,
    'starting_point_strategy': 'affine_step',
    'barrier_strategy': 'mehrotra_predictor_corrector',
    'use_line_search': False}

options = {
    'algorithm': 'mma'}

# Set up the optimizer
opt = ParOpt.Optimizer(topo, options)

# Set a new starting point
opt.optimize()
x, z, zw, zl, zu = opt.getOptimizedPoint()

