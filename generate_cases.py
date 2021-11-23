import numpy as np
from mpi4py import MPI
from tmr import TMR
import os
import pickle

def create_geo(AR, prob, forced_portion, MBB_bc_portion,
    ratio1, ratio2, use_hole, hole_radius):

    # Create the surface in the x-y plane
    Ly = 1.0
    Lx = Ly*AR
    nu = 2
    nv = 2
    x = np.linspace(0.0, Lx, nu)
    y = np.linspace(0.0, Ly, nv)
    pts = np.zeros((nu, nv, 3))
    for j in range(nv):
        for i in range(nu):
            pts[i,j,0] = x[i]
            pts[i,j,1] = y[j]

    tu = np.array([0.0, 0.0, Lx, Lx])
    tv = np.array([0.0, 0.0, Ly, Ly])

    # Create the b-spline surface
    surf = TMR.BsplineSurface(pts, tu=tu, tv=tv)
    face = TMR.FaceFromSurface(surf)

    if prob == 'cantilever':

        # Create the vertices on the surface
        v1 = TMR.VertexFromFace(face, 0.0, 0.0)
        v2 = TMR.VertexFromFace(face, Lx, 0.0)
        v3 = TMR.VertexFromFace(face, Lx, forced_portion*Ly)
        v4 = TMR.VertexFromFace(face, Lx, Ly)
        v5 = TMR.VertexFromFace(face, 0.0, Ly)
        verts = [v1, v2, v3, v4, v5]

        # Set up the edges
        pcurve1 = TMR.BsplinePcurve(np.array([[0.0, 0.0], [Lx, 0.0]]))
        pcurve2 = TMR.BsplinePcurve(np.array([[Lx, 0.0], [Lx, forced_portion*Ly]]))
        pcurve3 = TMR.BsplinePcurve(np.array([[Lx, forced_portion*Ly], [Lx, Ly]]))
        pcurve4 = TMR.BsplinePcurve(np.array([[Lx, Ly], [0.0, Ly]]))
        pcurve5 = TMR.BsplinePcurve(np.array([[0.0, Ly], [0.0, 0.0]]))

        edge1 = TMR.EdgeFromFace(face, pcurve1)
        edge2 = TMR.EdgeFromFace(face, pcurve2)
        edge3 = TMR.EdgeFromFace(face, pcurve3)
        edge4 = TMR.EdgeFromFace(face, pcurve4)
        edge5 = TMR.EdgeFromFace(face, pcurve5)

        edge1.setVertices(v1, v2)
        edge2.setVertices(v2, v3)
        edge3.setVertices(v3, v4)
        edge4.setVertices(v4, v5)
        edge5.setVertices(v5, v1)

        edge1.setName('1')
        edge2.setName('2')
        edge3.setName('3')
        edge4.setName('4')
        edge5.setName('5')

        edges = [edge1, edge2, edge3, edge4, edge5]
        dirs = [1, 1, 1, 1, 1]
        loop = TMR.EdgeLoop([edge1, edge2, edge3, edge4, edge5], dirs)
        face.addEdgeLoop(1, loop)

    elif prob == 'michell':

        # Create the vertices on the surface
        v1 = TMR.VertexFromFace(face, 0.0, 0.0)
        v2 = TMR.VertexFromFace(face, Lx, 0.0)
        v3 = TMR.VertexFromFace(face, Lx, 0.5*(1-forced_portion)*Ly)
        v4 = TMR.VertexFromFace(face, Lx, 0.5*(1+forced_portion)*Ly)
        v5 = TMR.VertexFromFace(face, Lx, Ly)
        v6 = TMR.VertexFromFace(face, 0.0, Ly)
        verts = [v1, v2, v3, v4, v5, v6]

        # Set up the edges
        pcurve1 = TMR.BsplinePcurve(np.array([[0.0, 0.0], [Lx, 0.0]]))
        pcurve2 = TMR.BsplinePcurve(np.array([[Lx, 0.0], [Lx, 0.5*(1-forced_portion)*Ly]]))
        pcurve3 = TMR.BsplinePcurve(np.array([[Lx, 0.5*(1-forced_portion)*Ly], [Lx, 0.5*(1+forced_portion)*Ly]]))
        pcurve4 = TMR.BsplinePcurve(np.array([[Lx, 0.5*(1+forced_portion)*Ly], [Lx, Ly]]))
        pcurve5 = TMR.BsplinePcurve(np.array([[Lx, Ly], [0.0, Ly]]))
        pcurve6 = TMR.BsplinePcurve(np.array([[0.0, Ly], [0.0, 0.0]]))

        edge1 = TMR.EdgeFromFace(face, pcurve1)
        edge2 = TMR.EdgeFromFace(face, pcurve2)
        edge3 = TMR.EdgeFromFace(face, pcurve3)
        edge4 = TMR.EdgeFromFace(face, pcurve4)
        edge5 = TMR.EdgeFromFace(face, pcurve5)
        edge6 = TMR.EdgeFromFace(face, pcurve6)

        edge1.setVertices(v1, v2)
        edge2.setVertices(v2, v3)
        edge3.setVertices(v3, v4)
        edge4.setVertices(v4, v5)
        edge5.setVertices(v5, v6)
        edge6.setVertices(v6, v1)

        edge1.setName('1')
        edge2.setName('2')
        edge3.setName('3')
        edge4.setName('4')
        edge5.setName('5')
        edge6.setName('6')

        edges = [edge1, edge2, edge3, edge4, edge5, edge6]
        dirs = [1, 1, 1, 1, 1, 1]
        loop = TMR.EdgeLoop([edge1, edge2, edge3, edge4, edge5, edge6], dirs)
        face.addEdgeLoop(1, loop)

    elif prob == 'MBB':

        # Create the vertices on the surface
        v1 = TMR.VertexFromFace(face, 0.0, 0.0)
        v2 = TMR.VertexFromFace(face, Lx*(1-MBB_bc_portion), 0.0)
        v3 = TMR.VertexFromFace(face, Lx, 0.0)
        v4 = TMR.VertexFromFace(face, Lx, Ly)
        v5 = TMR.VertexFromFace(face, Lx*forced_portion, Ly)
        v6 = TMR.VertexFromFace(face, 0.0, Ly)
        verts = [v1, v2, v3, v4, v5, v6]

        # Set up the edges
        pcurve1 = TMR.BsplinePcurve(np.array([[0.0, 0.0], [Lx*(1-MBB_bc_portion), 0.0]]))
        pcurve2 = TMR.BsplinePcurve(np.array([[Lx*(1-MBB_bc_portion), 0.0], [Lx, 0.0]]))
        pcurve3 = TMR.BsplinePcurve(np.array([[Lx, 0.0], [Lx, Ly]]))
        pcurve4 = TMR.BsplinePcurve(np.array([[Lx, Ly], [Lx*forced_portion, Ly]]))
        pcurve5 = TMR.BsplinePcurve(np.array([[Lx*forced_portion, Ly], [0.0, Ly]]))
        pcurve6 = TMR.BsplinePcurve(np.array([[0.0, Ly], [0.0, 0.0]]))

        edge1 = TMR.EdgeFromFace(face, pcurve1)
        edge2 = TMR.EdgeFromFace(face, pcurve2)
        edge3 = TMR.EdgeFromFace(face, pcurve3)
        edge4 = TMR.EdgeFromFace(face, pcurve4)
        edge5 = TMR.EdgeFromFace(face, pcurve5)
        edge6 = TMR.EdgeFromFace(face, pcurve6)

        edge1.setVertices(v1, v2)
        edge2.setVertices(v2, v3)
        edge3.setVertices(v3, v4)
        edge4.setVertices(v4, v5)
        edge5.setVertices(v5, v6)
        edge6.setVertices(v6, v1)

        edge1.setName('1')
        edge2.setName('2')
        edge3.setName('3')
        edge4.setName('4')
        edge5.setName('5')
        edge6.setName('6')

        edges = [edge1, edge2, edge3, edge4, edge5, edge6]
        dirs = [1, 1, 1, 1, 1, 1]
        loop = TMR.EdgeLoop([edge1, edge2, edge3, edge4, edge5, edge6], dirs)
        face.addEdgeLoop(1, loop)

    elif prob == 'lbracket':

        # Create the vertices on the surface
        v1 = TMR.VertexFromFace(face, 0.0, 0.0)
        v2 = TMR.VertexFromFace(face, Lx, 0.0)
        v3 = TMR.VertexFromFace(face, Lx, Ly*ratio2*(1-forced_portion))
        v4 = TMR.VertexFromFace(face, Lx, Ly*ratio2)
        v5 = TMR.VertexFromFace(face, Lx*ratio1, Ly*ratio2)
        v6 = TMR.VertexFromFace(face, Lx*ratio1, Ly)
        v7 = TMR.VertexFromFace(face, 0.0, Ly)
        verts = [v1, v2, v3, v4, v5, v6, v7]

        # Set up the edges
        pcurve1 = TMR.BsplinePcurve(np.array([[0.0, 0.0], [Lx, 0.0]]))
        pcurve2 = TMR.BsplinePcurve(np.array([[Lx, 0.0], [Lx, Ly*ratio2*(1-forced_portion)]]))
        pcurve3 = TMR.BsplinePcurve(np.array([[Lx, Ly*ratio2*(1-forced_portion)], [Lx, Ly*ratio2]]))
        pcurve4 = TMR.BsplinePcurve(np.array([[Lx, Ly*ratio2], [Lx*ratio1, Ly*ratio2]]))
        pcurve5 = TMR.BsplinePcurve(np.array([[Lx*ratio1, Ly*ratio2], [Lx*ratio1, Ly]]))
        pcurve6 = TMR.BsplinePcurve(np.array([[Lx*ratio1, Ly], [0.0, Ly]]))
        pcurve7 = TMR.BsplinePcurve(np.array([[0.0, Ly], [0.0, 0.0]]))

        edge1 = TMR.EdgeFromFace(face, pcurve1)
        edge2 = TMR.EdgeFromFace(face, pcurve2)
        edge3 = TMR.EdgeFromFace(face, pcurve3)
        edge4 = TMR.EdgeFromFace(face, pcurve4)
        edge5 = TMR.EdgeFromFace(face, pcurve5)
        edge6 = TMR.EdgeFromFace(face, pcurve6)
        edge7 = TMR.EdgeFromFace(face, pcurve7)

        edge1.setVertices(v1, v2)
        edge2.setVertices(v2, v3)
        edge3.setVertices(v3, v4)
        edge4.setVertices(v4, v5)
        edge5.setVertices(v5, v6)
        edge6.setVertices(v6, v7)
        edge7.setVertices(v7, v1)

        edge1.setName('1')
        edge2.setName('2')
        edge3.setName('3')
        edge4.setName('4')
        edge5.setName('5')
        edge6.setName('6')
        edge7.setName('7')

        edges = [edge1, edge2, edge3, edge4, edge5, edge6, edge7]
        dirs = [1, 1, 1, 1, 1, 1, 1]
        loop = TMR.EdgeLoop([edge1, edge2, edge3, edge4, edge5, edge6, edge7], dirs)
        face.addEdgeLoop(1, loop)

    # Set up the hole
    if use_hole:
        if prob == 'lbracket':
            r = hole_radius*Ly*ratio2
            xc = 0.5*(1+ratio1)*Lx
            yc = 0.5*Ly*ratio2

        else:
            r = hole_radius *Ly
            xc = 0.5*Lx
            yc = 0.5*Ly

        vc = TMR.VertexFromFace(face, xc - r, yc)

        pts = [[-r, 0.0], [-r, r], [0.0, r], [r, r],
            [r, 0.0], [r, -r], [0.0, -r], [-r, -r], [-r, 0.0]]
        pts = np.array(pts)
        for i in range(pts.shape[0]):
            pts[i,0] += xc
            pts[i,1] += yc

        wts = [1.0, 1.0/np.sqrt(2), 1.0, 1.0/np.sqrt(2),
            1.0, 1.0/np.sqrt(2), 1.0, 1.0/np.sqrt(2), 1.0]
        Tu = [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]

        pcurvec = TMR.BsplinePcurve(np.array(pts),
            tu=np.array(Tu), wts=np.array(wts), k=3)

        edgec = TMR.EdgeFromFace(face, pcurvec)
        edgec.setVertices(vc, vc)

        loop = TMR.EdgeLoop([edgec], [1])
        face.addEdgeLoop(-1, loop)

        verts.append(vc)
        edges.append(edgec)


    # Create the TMRModel
    faces = [face]
    geo = TMR.Model(verts, edges, faces)

    return geo

def create_mesh(n, AR, prob, ratio1, ratio2, forced_portion, MBB_bc_portion,
    force_magnitude, use_hole, hole_radius):

    Ly = 1.0
    Lx = Ly*AR

    # Create tmr geometry object
    geo = create_geo(AR, prob, forced_portion, MBB_bc_portion,
        ratio1, ratio2, use_hole, hole_radius)

    # Create the mesh
    comm = MPI.COMM_WORLD
    mesh = TMR.Mesh(comm, geo)

    # Mesh the part
    opts = TMR.MeshOptions()
    opts.frontal_quality_factor = 1.25
    opts.num_smoothing_steps = 20
    opts.write_mesh_quality_histogram = 1

    # Mesh the geometry with the given target size
    htarget = Ly / n
    mesh.mesh(htarget, opts=opts)

    # Create a model from the mesh
    model = mesh.createModelFromMesh()

    # Create the corresponding mesh topology from the mesh-model
    topo = TMR.Topology(comm, model)

    # Create the quad forest and set the topology of the forest
    forest = TMR.QuadForest(comm)
    forest.setTopology(topo)

    # Create random trees and balance the mesh. Print the output file
    forest.createTrees()
    forest.balance(1)

    # Create the nodes
    forest.createNodes()

    # Get the mesh connectivity
    conn = forest.getMeshConn()

    # Get the node locations
    X = forest.getPoints()

    # Set the nodal positions using only the x and y locations
    X = np.array(X[:,:2])

    # Get the nodes with the specified name
    if prob == 'cantilever':
        bcns = forest.getNodesWithName('5')
    elif prob == 'MBB':
        bcns1 = forest.getNodesWithName('6')
        bcns2 = forest.getNodesWithName('2')
    elif prob == 'michell':
        bcns = forest.getNodesWithName('6')
    elif prob == 'lbracket':
        bcns = forest.getNodesWithName('6')

    # Create the vars array with the number of nodes
    nnodes = np.max(conn)+1

    # Set the vars to a negative index where we have a constraint
    bcs = {}
    if prob == 'MBB':

        for node in bcns1:
            bcs[node] = [0]
        for node in bcns2:
            bcs[node] = [0, 1]

    else:
        for node in bcns:
            bcs[node] = [0, 1]

    # Set up the forces
    forces = {}

    # Assign the force vector
    if prob == 'cantilever':

        south_east_corner_node = -1
        xpos = X[0, 0]
        ypos = X[0, 0]
        for i in range(nnodes):
            if X[i, 0] >= xpos and X[i, 1] <= ypos:
                south_east_corner_node = i
                xpos = X[i, 0]
                ypos = X[i, 1]

        forces[south_east_corner_node] = [0.0, -force_magnitude]

    elif prob == 'michell':

        distance = Lx**2 + Ly**2
        xtarget = Lx
        ytarget = Ly / 2
        middle_node = -1
        for i in range(nnodes):
            xpos = X[i, 0]
            ypos = X[i, 1]
            if (xpos-xtarget)**2 + (ypos-ytarget)**2 <= distance:
                middle_node = i
                distance = (xpos-xtarget)**2 + (ypos-ytarget)**2

        forces[middle_node] = [0.0, -force_magnitude]

    elif prob == 'MBB':

        distance = Lx**2 + Ly**2
        xtarget = 0
        ytarget = Ly
        north_west_node = -1
        for i in range(nnodes):
            xpos = X[i, 0]
            ypos = X[i, 1]
            if (xpos-xtarget)**2 + (ypos-ytarget)**2 <= distance:
                north_west_node = i
                distance = (xpos-xtarget)**2 + (ypos-ytarget)**2

        forces[north_west_node] = [0.0, -force_magnitude]

    elif prob == 'lbracket':

        distance = Lx**2 + Ly**2
        xtarget = Lx
        ytarget = Ly*ratio2
        middle_node = -1
        for i in range(nnodes):
            xpos = X[i, 0]
            ypos = X[i, 1]
            if (xpos-xtarget)**2 + (ypos-ytarget)**2 <= distance:
                middle_node = i
                distance = (xpos-xtarget)**2 + (ypos-ytarget)**2

        forces[middle_node] = [0.0, -force_magnitude]

    nelems = len(conn)

    return nelems, nnodes, conn, X, bcs, forces


if __name__ == '__main__':
    prefix = 'cases'

    AR = 1.0
    ratio1 = 0.4
    ratio2 = 0.4
    forced_portion = 0.2
    force_magnitude = 25.0
    MBB_bc_portion = 0.1
    hole_radius = 0.1

    for prob in ['lbracket', 'cantilever', 'michell']:
        for n in [32, 64, 96]:
            for use_hole in [ True, False ]:
                if prob == 'cantilever':
                    AR = 3

                nelems, nnodes, conn_tmr, X, bcs, forces = create_mesh(
                    n, AR, prob, ratio1, ratio2, forced_portion,
                    MBB_bc_portion, force_magnitude,
                    use_hole, hole_radius)

                conn = conn_tmr.copy()
                conn[:,2] = conn_tmr[:,3]
                conn[:,3] = conn_tmr[:,2]

                pkl = dict()
                pkl['n'] = n
                pkl['nelems'] = nelems
                pkl['nelems'] = nelems
                pkl['nnodes'] = nnodes
                pkl['conn'] = conn
                pkl['X'] = X
                pkl['bcs'] = bcs
                pkl['forces'] = forces

                filename = os.path.join(prefix, '%s_n=%d'%(prob, n))
                if use_hole:
                    filename += '_hole.pkl'
                else:
                    filename += '.pkl'

                with open(filename, 'wb') as pklfile:
                    pickle.dump(pkl, pklfile)