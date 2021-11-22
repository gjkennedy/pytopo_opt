import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.optimize import minimize

class Truss:
    def __init__(self, conn, x, bcs, f, E=10.0, rho=0.1):
        self.conn = np.array(conn, dtype=int)
        self.x = np.array(x)
        self.bcs = np.array(bcs, dtype=int)
        self.f = np.array(f)
        self.E = E
        self.rho = rho

        # Extract the number of nodes and elements
        self.nnodes = self.x.shape[0]
        self.nelems = self.conn.shape[0]

        # Form the reduced set of degrees of freedom
        self.reduced = np.setdiff1d(np.arange(2*self.nnodes), self.bcs)

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.elem_vars = np.array((2*self.conn[:,0],
                                   2*self.conn[:,0]+1,
                                   2*self.conn[:,1],
                                   2*self.conn[:,1]+1)).T
        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.elem_vars[index, :]:
                for jj in self.elem_vars[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)

        return

    def assemble_stiffness_matrix(self, A):
        """
        Assemble the stiffness matrix
        """

        # Compute the x and y distances between nodes
        xd = self.x[self.conn[:,1],0] - self.x[self.conn[:,0],0]
        yd = self.x[self.conn[:,1],1] - self.x[self.conn[:,0],1]
        Le = np.sqrt(xd**2 + yd**2)
        C = xd/Le
        S = yd/Le

        # Compute the B matrix
        B = np.zeros((self.nelems, 4))
        B[:, 0] = -C/Le
        B[:, 1] = -S/Le
        B[:, 2] = C/Le
        B[:, 3] = S/Le

        scale = self.E*A*Le

        # Compute all of the element matrices
        Ke = np.zeros((self.nelems, 4, 4), dtype=A.dtype)
        for i in range(self.nelems):
            Ke[i,:] = scale[i]*np.outer(B[i,:], B[i,:])

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def compute_stiffness_matrix_derivative(self, y, z=None, alpha=1.0):
        """
        Compute the result:

        product = alpha * d(y^{T}*K(x)*z)/dx

        If z is None then the code uses y = z.
        """

        # Compute the x and y distances between nodes
        xd = self.x[self.conn[:,1],0] - self.x[self.conn[:,0],0]
        yd = self.x[self.conn[:,1],1] - self.x[self.conn[:,0],1]
        Le = np.sqrt(xd**2 + yd**2)
        C = xd/Le
        S = yd/Le

        # Compute the B matrix
        B = np.zeros((self.nelems, 4))
        B[:, 0] = -C/Le
        B[:, 1] = -S/Le
        B[:, 2] = C/Le
        B[:, 3] = S/Le

        scale = self.E*Le

        if z is None:
            y_e = y[self.elem_vars]
            strain_y = np.sum(B*y_e, axis=1)

            product = alpha*scale*strain_y**2
        else:
            y_e = y[self.elem_vars]
            strain_y = np.sum(B*y_e, axis=1)

            z_e = z[self.elem_vars]
            strain_z = np.sum(B*z_e, axis=1)

            product = alpha*scale*strain_y*strain_z

        return product

    def assemble_mass_matrix(self, A):
        """
        Assemble the mass matrix
        """

        # Compute the x and y distances between nodes
        xd = self.x[self.conn[:,1],0] - self.x[self.conn[:,0],0]
        yd = self.x[self.conn[:,1],1] - self.x[self.conn[:,0],1]
        Le = np.sqrt(xd**2 + yd**2)
        C = xd/Le
        S = yd/Le

        scale = 0.5*self.rho*A*Le

        N = np.zeros((self.nelems, 4))
        pt = 1.0/np.sqrt(3.0)
        u = 0.5 - 0.5*pt
        N[:, 0] = C*(1.0 - u)
        N[:, 1] = S*(1.0 - u)
        N[:, 2] = C*u
        N[:, 3] = S*u

        Me = np.zeros((self.nelems, 4, 4), dtype=A.dtype)
        for i in range(self.nelems):
            Me[i,:] += scale[i]*np.outer(N[i,:], N[i,:])

        u = 0.5 + 0.5*pt
        N[:, 0] = C*(1.0 - u)
        N[:, 1] = S*(1.0 - u)
        N[:, 2] = C*u
        N[:, 3] = S*u

        for i in range(self.nelems):
            Me[i,:] += scale[i]*np.outer(N[i,:], N[i,:])

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        return M

    def compute_mass_matrix_derivative(self, y, z=None, alpha=1.0):
        """
        Compute the result:

        product = alpha * d(y^{T}*M(x)*z)/dx

        If z is None then the code uses y = z.
        """

        # Compute the x and y distances between nodes
        xd = self.x[self.conn[:,1],0] - self.x[self.conn[:,0],0]
        yd = self.x[self.conn[:,1],1] - self.x[self.conn[:,0],1]
        Le = np.sqrt(xd**2 + yd**2)
        C = xd/Le
        S = yd/Le


        scale = 0.5*self.rho*Le

        N = np.zeros((self.nelems, 4))
        pt = 1.0/np.sqrt(3.0)
        u = 0.5 - 0.5*pt
        N[:, 0] = C*(1.0 - u)
        N[:, 1] = S*(1.0 - u)
        N[:, 2] = C*u
        N[:, 3] = S*u

        if z is None:
            y_e = y[self.elem_vars]
            Ny = np.sum(N*y_e, axis=1)

            product = alpha*scale*Ny**2
        else:
            y_e = y[self.elem_vars]
            Ny = np.sum(N*y_e, axis=1)

            z_e = z[self.elem_vars]
            Nz = np.sum(N*z_e, axis=1)

            product = alpha*scale*Ny*Nz

        u = 0.5 + 0.5*pt
        N[:, 0] = C*(1.0 - u)
        N[:, 1] = S*(1.0 - u)
        N[:, 2] = C*u
        N[:, 3] = S*u

        if z is None:
            y_e = y[self.elem_vars]
            Ny = np.sum(N*y_e, axis=1)

            product += alpha*scale*Ny**2
        else:
            y_e = y[self.elem_vars]
            Ny = np.sum(N*y_e, axis=1)

            z_e = z[self.elem_vars]
            Nz = np.sum(N*z_e, axis=1)

            product += alpha*scale*Ny*Nz

        return product

    def compliance(self, A):
        """
        Given the cross-sectional areas, compute the compliance
        """

        u = self.solve(A)
        return np.dot(u, self.f)

    def compliance_gradient(self, A):
        """
        Given the cross-sectional areas, compute the compliance gradient
        """

        u = self.solve(A)
        return self.compute_stiffness_matrix_derivative(u, alpha=-1.0)

    def frequencies(self, A, k=5, sigma=0.0):
        """
        Compute the k-th smallest natural frequencies
        """

        K = self.assemble_stiffness_matrix(A)
        Kr = self.reduce_matrix(K)

        M = self.assemble_mass_matrix(A)
        Mr = self.reduce_matrix(M)

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        if k < len(self.reduced):
            eigs, phir = sparse.linalg.eigsh(Kr, M=Mr, k=k, sigma=sigma,
                                             which='LM', tol=1e-6)
        else:
            eigs, phir = scipy.linalg.eigh(Kr.todense(), b=Mr.todense())
            k = len(eigs)

        phi = np.zeros((2*self.nnodes, k))
        for i in range(k):
            phi[self.reduced, i] = phir[:, i]

        return np.sqrt(eigs), phi

    def frequency_derivative(self, A, k=5):
        """
        Compute the gradient of the smallest eigenvalues, assuming they are unique
        """
        if k > len(self.reduced):
            k = len(self.reduced)

        omega, phi = self.frequencies(A, k=k)

        omega_grad = []
        for i in range(k):
            kx = self.compute_stiffness_matrix_derivative(phi[:, i])
            mx = self.compute_mass_matrix_derivative(phi[:, i])
            grad = kx - mx*omega[i]**2
            omega_grad.append((0.5/omega[i])*grad)

        return omega_grad

    def ks_min_eigenvalue(self, A, ks_rho=100.0, k=5):
        """
        Compute the ks minimum eigenvalue
        """
        if k > len(self.reduced):
            k = len(self.reduced)

        omega, phi = self.frequencies(A, k=k)
        lamb = omega**2

        c = np.min(lamb)
        eta = np.exp(-ks_rho*(lamb - c))
        a = np.sum(eta)
        ks_min = c - np.log(a)/ks_rho
        eta *= 1.0/a

        ks_grad = np.zeros(self.nelems)
        for i in range(k):
            kx = self.compute_stiffness_matrix_derivative(phi[:, i])
            mx = self.compute_mass_matrix_derivative(phi[:, i])
            ks_grad += eta[i]*(kx - mx*lamb[i])

        return ks_min, ks_grad

    def eigenvector_derivative(self, A, k=5, i=0):

        if k > len(self.reduced):
            k = len(self.reduced)

        omega, phi = self.frequencies(A, k=k)
        lamb = omega**2

        mx = self.compute_mass_matrix_derivative(phi[:, i])
        grad = -0.5*np.outer(phi[:, i], mx)

        for j in range(k):
            if j != i:
                kx = self.compute_stiffness_matrix_derivative(phi[:, i], phi[:, j])
                mx = self.compute_mass_matrix_derivative(phi[:, i], phi[:, j])
                gx = (kx - lamb[i]*mx)/(lamb[i] - lamb[j])
                grad += np.outer(phi[:, j], gx)

        return grad

    def compute_stresses(self, u):
        """
        Compute the stresses in each element
        """

        # Compute the x and y distances between nodes
        xd = self.x[self.conn[:,1],0] - self.x[self.conn[:,0],0]
        yd = self.x[self.conn[:,1],1] - self.x[self.conn[:,0],1]
        Le = np.sqrt(xd**2 + yd**2)
        C = xd/Le
        S = yd/Le

        # Compute the B matrix
        B = np.zeros((self.nelems, 4))
        B[:, 0] = -C/Le
        B[:, 1] = -S/Le
        B[:, 2] = C/Le
        B[:, 3] = S/Le

        ue = np.array((u[2*self.conn[:,0]],
                       u[2*self.conn[:,0]+1],
                       u[2*self.conn[:,1]],
                       u[2*self.conn[:,1]+1])).T

        stress = self.E*np.sum(B*ue, axis=1)
        return stress

    def compute_mass_gradient(self):

        # Compute the gradient of the mass of the truss
        xd = self.x[truss.conn[:,1],0] - self.x[truss.conn[:,0],0]
        yd = self.x[truss.conn[:,1],1] - self.x[truss.conn[:,0],1]
        Le = np.sqrt(xd**2 + yd**2)

        return self.rho*Le

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

    def solve(self, A):
        """
        Perform a linear static analysis
        """

        K = self.assemble_stiffness_matrix(A)
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.f)

        ur = sparse.linalg.spsolve(Kr, fr)

        u = np.zeros(2*self.nnodes, dtype=ur.dtype)
        u[self.reduced] = ur

        return u

    def plot(self, u=None, scale=1.0, **kwargs):
        """
        Visualize the truss and optionally its deformation.
        """
        fig, ax = plt.subplots(1, 1, facecolor='w', **kwargs)

        if u is not None:
            x = self.x + scale*u.reshape((-1, 2))
        else:
            x = self.x

        for index in range(self.nelems):
            i = self.conn[index, 0]
            j = self.conn[index, 1]
            plt.plot([x[i,0], x[j,0]], [x[i,1], x[j,1]], '-ko')

        ax.axis('equal')

        return

    def plot_areas(self, A, **kwargs):
        """
        Plot the bar areas
        """
        fig, ax = plt.subplots(1, 1, facecolor='w', **kwargs)

        stress = self.compute_stresses(self.solve(A))

        cm = plt.get_cmap('coolwarm')
        cNorm = colors.Normalize(vmin=min(stress), vmax=max(stress))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        for index, bar in enumerate(conn):
            n1 = bar[0]
            n2 = bar[1]
            if A[index] >= 1e-4*max(A):
                plt.plot([self.x[n1, 0], self.x[n2, 0]],
                         [self.x[n1, 1], self.x[n2, 1]],
                         color=scalarMap.to_rgba(stress[index]),
                         linewidth=5*(A[index]/max(A)))

        ax.axis('equal')

        return

E = 70e9
rho = 2700.0

# Set the parameters for the truss optimization
L = 2.5
P = 250e3

# Set the connectivity of the truss
M = 6
conn = []
for i in range(M-1):
    conn.extend([[2*i, 2*(i+1)],
                 [2*i, 2*i+1],
                 [2*i, 2*(i+1)+1],
                 [2*i+1, 2*(i+1)],
                 [2*i+1, 2*(i+1)+1]])
conn.append([2*(M-1), 2*(M-1)+1])

# Set the positions
pos = []
for i in range(M):
    u = 1.0*i/(M-1.0)
    pos.append([i*L, 0.0])
    pos.append([i*L, L])

# Set the boundary conditions
bcs = [0, 1,
       2*(2*(M-1)) + 1]

# Apply the loads
force = [0]*(2*len(pos))
for i in range(M):
    force[4*i + 1] = -P

truss = Truss(conn, pos, bcs, force, E=E, rho=rho)

m_fixed = (2*M*np.sqrt(2) + 4*M)*L*rho

# Set the initial bar areas
x0 = np.ones(len(conn))

Ascale = 1e-3
objscale = 1e5

objective = lambda x : truss.compliance(Ascale*x)/objscale
gradient = lambda x : truss.compliance_gradient(Ascale*x)*Ascale/objscale

epsilon = 1e-6
p = np.random.uniform(size=x0.shape)
fd = (objective(x0 + epsilon*p) - objective(x0))/epsilon
result = np.dot(p, gradient(x0))

print('Objective check')
print(objective(x0))
print(fd)
print(result)

m0 = truss.compute_mass_gradient()
mass_fixed = 1e-3*np.sum(m0)

mass_con = lambda x : 1.0 - np.dot(m0, Ascale*x)/mass_fixed
mass_con_gradient = lambda x : - Ascale*m0/mass_fixed

fd = (mass_con(x0 + epsilon*p) - mass_con(x0))/epsilon
result = np.dot(p, mass_con_gradient(x0))

print('Mass constraint check')
print('fixed mass = ', mass_fixed)
print(mass_con(x0))
print(fd)
print(result)

omega0 = 500.0

freq_con = lambda x : truss.ks_min_eigenvalue(Ascale*x)[0]/omega0**2 - 1.0
freq_con_gradient = lambda x : Ascale*truss.ks_min_eigenvalue(Ascale*x)[1]/omega0**2

fd = (freq_con(x0 + epsilon*p) - freq_con(x0))/epsilon
result = np.dot(p, freq_con_gradient(x0))

print('KS constraint check')
print(freq_con(x0))
print(fd)
print(result)

A_min = 1e-5
A_max = 10.0

x_min = A_min/Ascale
x_max = A_max/Ascale

x0 = x_min*np.ones(x0.shape)
# x0 = x_min + (x_max - x_min)*np.random.uniform(size=len(conn))

res = minimize(objective, x0, jac=gradient, options={'maxiter' : 500},
               method='SLSQP',
               bounds=[(x_min, x_max)]*len(x0),
               constraints=[{'type': 'ineq',
                             'fun': mass_con,
                             'jac': mass_con_gradient},
                             {'type': 'ineq',
                             'fun': freq_con,
                             'jac': freq_con_gradient}
                             ])
print(res)

Astar = Ascale*res.x
print('Design variables: ', res.x)
print('Bar areas:        ', Ascale*res.x)

print(truss.frequencies(Astar)[0])

truss.plot_areas(Astar)
plt.show()