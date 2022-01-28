import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffpy.Structure import loadStructure, Lattice
from diffpy.srreal.overlapcalculator import OverlapCalculator
import os

def define_space(stru, path, nAtoms=100, plot=False):
    bc = OverlapCalculator()
    cluster = loadStructure(path +'/'+ stru)
    bc(cluster)
    if len(cluster) < 4 or nAtoms < len(cluster):
        return None
    else:
        #print(stru, len(cluster))
        pass

    atoms = np.array([cluster.x,cluster.y,cluster.z]).T
    ph_atoms = atoms
    A = atoms[0]
    B = atoms[1]
    atoms = center_atom(A, atoms)

    stru_dim = None
    for i, cords in enumerate(atoms[2:]):  # A, B and C cant be on the same line
        C = cords
        on_line = line_check_3d(A, B, C)
        if on_line != True:
            break
        elif i == len(atoms)-3:
            print('Molecule is 1D!')
            C = atoms[2]  # Molecule is linear
              # Move atoms so that label1 is origo
            stru_dim = '1D'
            new_x = unit_vector(B - A)
            atoms, new_x, new_y, new_z = align_mat(atoms, new_x, [None], [None])
            new_y = [0,1,0]
            new_z = [0, 0, 1]
            break
        else:
            continue

    for i, cords in enumerate(atoms[2:]):  # A, B and C cant be on the same line
        if stru_dim == '1D':
            break

        if np.array_equal(C,cords) == False:  # D and C cant be the same atom
            D = cords
            n, d = get_plane(A, B, C)  # Get plane for A, B, C
            in_plane = (n[0] * D[0] + n[1] * D[1] + n[2] * D[2] + d == 0)
        else:
            in_plane = None
            pass
        if in_plane == False:
            break
        elif i == len(atoms)-3:
            print('Molecule is 2D!')
            stru_dim = '2D'
            new_x = unit_vector(B - A)
            new_z = unit_vector(n)
            n, d = get_plane(A, new_x, new_z)
            new_y = unit_vector(n)

            new_y = direction_check(C, new_y)
            new_z = direction_check(D, new_z)
            atoms, new_x, new_y, new_z = align_mat(atoms, np.array(new_x), np.array(new_y), np.array(new_z))
            break
        else:
            continue



    if stru_dim == None:
        #print('Molecule is 3D!')
        stru_dim = '3D'
        new_x = unit_vector(B-A)
        new_z = unit_vector(n)
        n, d = get_plane(A, new_x, new_z)
        new_y = unit_vector(n)

        new_y = direction_check(C, new_y)
        new_z = direction_check(D, new_z)

        atoms, new_x, new_y, new_z = align_mat(atoms, new_x, new_y, new_z)


    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([new_x[0]*-10, 0], [new_x[1]*-10, 0], [new_x[2]*-10, 0], label='-x')
        ax.plot([0, new_x[0] * 10], [0, new_x[1] * 10], [0, new_x[2] * 10], label='+x')
        ax.plot([new_y[0]*-10, 0], [new_y[1]*-10, 0], [new_y[2]*-10, 0], label='-y')
        ax.plot([0, new_y[0] * 10], [0, new_y[1] * 10], [0, new_y[2] * 10],label='+y')
        ax.plot([new_z[0]*-10, 0], [new_z[1]*-10, 0], [new_z[2]*-10, 0], label='-z')
        ax.plot([0, new_z[0]*10], [0, new_z[1]*10], [0, new_z[2]*10], label='+z')

        for i, cords in enumerate(atoms):
            ax.scatter(cords[0], cords[1], cords[2], marker='o', label=i)

        for i, cords in enumerate(ph_atoms):
            ax.scatter(cords[0], cords[1], cords[2], marker='d', label=i)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.legend()
        plt.show()
        fig.clf()

    atoms = atoms.T
    cluster.x = atoms[0]
    cluster.y = atoms[1]
    cluster.z = atoms[2]

    cluster.write(path +'/'+ stru, format="xyz")

    return cluster, stru_dim

def center_atom(center,atoms):
    x = center[0]
    y = center[1]
    z = center[2]

    atoms = atoms.T
    atoms[0] -= x
    atoms[1] -= y
    atoms[2] -= z

    atoms = atoms.T
    return atoms

#def normFeatures(x, min, max):
#    return 1*((x-min)/(max-min))-0



def align_mat(atoms, x,y,z, info=False):
    if info:
        print(x,y,z)
    if y[0] == None and z[0] == None:
        rot = rotation_matrix(x, [1, 0, 0])
        x = rot.dot(x)
        ph = []
        for cords in atoms:
            ph.append(rot.dot(cords))
        atoms = np.array(ph)
        return atoms, x, y, z

    if np.array_equiv(-x, [1., 0., 0.]) == True and np.array_equiv(abs(y), [0.,1.,0.]) == True and np.array_equiv(abs(z), [0.,0.,1.]) == True:
        if info:
            print('x inversion')
        x[0] *= -1
        y[0] *= -1
        z[0] *= -1
        atoms = atoms.T
        atoms[0] *= -1
        atoms = atoms.T
    if np.array_equiv(abs(x), [1., 0., 0.]) == True and np.array_equiv(-y, [0.,1.,0.]) == True and np.array_equiv(abs(z), [0.,0.,1.]) == True:
        if info:
            print('y inversion')
        x[1] *= -1
        y[1] *= -1
        z[1] *= -1
        atoms = atoms.T
        atoms[1] *= -1
        atoms = atoms.T
    if np.array_equiv(-abs(x), [1., 0., 0.]) == True and np.array_equiv(abs(y), [0.,1.,0.]) == True and np.array_equiv(-z, [0.,0.,1.]) == True:
        if info:
            print('x inversion')
        x[2] *= -1
        y[2] *= -1
        z[2] *= -1
        atoms = atoms.T
        atoms[2] *= -1
        atoms = atoms.T

    ### Rotation
    if np.array_equiv(-x,[1,0,0]) == True:
        if info:
            print('180 x')
        x[:2] *= -1
        y[:2] *= -1
        z[:2] *= -1

        atoms = atoms.T
        atoms[0] *= -1
        atoms[1] *= -1
        atoms = atoms.T
    if np.array_equiv(-y,[0,1,0]) == True:
        if info:
            print('180 y')
        x[:2] *= -1
        y[:2] *= -1
        z[:2] *= -1

        atoms = atoms.T
        atoms[1] *= -1
        atoms[2] *= -1
        atoms = atoms.T
    if np.array_equiv(-z,[0,0,1]) == True:
        if info:
            print('180 z')
        x[1:] *= -1
        y[1:] *= -1
        z[1:] *= -1

        atoms = atoms.T
        atoms[0] *= -1
        atoms[2] *= -1
        atoms = atoms.T

    if np.array_equiv(x,[1,0,0]) == False:
        rot = rotation_matrix(x, [1, 0, 0])
        x = rot.dot(x)
        y = rot.dot(y)
        z = rot.dot(z)

        ph = []
        for cords in atoms:
            ph.append(rot.dot(cords))
        atoms = np.array(ph)

    if np.array_equiv(abs(y),[0,1,0]) == False:
        rot = rotation_matrix(y, [0, 1, 0])
        y = rot.dot(y)
        z = rot.dot(z)
        ph = []
        for cords in atoms:
            ph.append(rot.dot(cords))
        atoms = np.array(ph)

    return atoms,x,y,z


def rotation_matrix(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def direction_check(point, vector):
    dist1 = np.linalg.norm(point - vector)
    dist2 = np.linalg.norm(point - (-vector))
    if dist1 > dist2:
        return -vector
    else:
        return vector
    return vector

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def get_plane(A, B, C):
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)
    d = -A.dot(n)

    return n, d

def line_check_3d(A,B,C):
    v = B - A
    u = C - A
    v = unit_vector(v)
    u = unit_vector(u)
    l = abs(u.dot(v))
    l = '{:.6f}'.format(l)

    if l >= '{:.6f}'.format(0.98):
        return True
    else:
        return False


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return (rad/np.pi)*180


if __name__ == '__main__':
    """
    Notes:
        1) Should have a function that normilizes xyz
    """
    folder = '/home/skaaning/Graph_Creator/MetalClusters'
    enliste = os.listdir(folder)
    enliste = [file for file in enliste if file[0] != '.' and file[-4:] == '.xyz']
    enliste = sorted(enliste[:2])

    satellites = [[1,0,0],  # Must be between -1 and 1
                  [0,1,0],
                  [0,0,1],
                  [-0.3,-0.3,-0.3]]

    atoms = np.array([[0.1,0.1,0.1],
                      [-2.2,0,0],
                      [-2.1,-0.2,0],
                      [-2,0,-0.3]])

    for stru in enliste:
        cluster = define_space(stru, folder)  # All distances must be calculated first
