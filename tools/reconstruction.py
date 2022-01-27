import scipy, sys, h5py, os, pdb
from scipy.optimize import least_squares, minimize, fmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from mendeleev import element
from tqdm import tqdm
#from norm_space import define_space
np.random.seed(12)

def read_h5py(file, path):
    data = []
    labels = []

    with h5py.File(path+'/'+file, 'r') as f:
        # List all groups
        #print("Keys: %s" % f.keys())
        data_ph = []
        keys = list(f.keys())
        labels.append(keys)
        for i in keys:
            for j in range(len(f[i])):
                data_ph.append(list(f[i][j]))
    data.append(data_ph)

    return np.array(data), labels


def gen_CIF(atoms, cords, path, name = 'test.xyz'):
    f = open(path + '/gen_XYZ/' + name, "w")

    cords[:,0] = cords[:,0] - cords[:,0].mean()
    cords[:,1] = cords[:,1] - cords[:,1].mean()
    cords[:,2] = cords[:,2] - cords[:,2].mean()

    for i in range(len(atoms)):
        if i == 0:
            f.write('{}\n\n'.format(len(atoms)))
        f.write('{:3s} {:>.20f} {:>.20f} {:>.20f}\n'.format(atoms[i],
                                                          cords[i][0],
                                                          cords[i][1],
                                                          cords[i][2]))
    f.close()
    #define_space("gen_XYZ/"+name, path, plot=False)

    return None

def renormalise(norm_x, xmin, xmax):
    y = ((norm_x + 0) * (xmax - xmin) / 1) + xmin
    return y

# Define a function that evaluates the equations
def equations(guess, satellites):
    x, y, z = guess  # r is set to 0 in equation

    #r = 0

    mse = 0.0
    for cords in satellites:

        mse += abs(np.sqrt((x - cords[0]) ** 2 + (y - cords[1]) ** 2 + (z - cords[2]) ** 2) - cords[3])
        history.append([mse, x, y, z])
    return mse #/ len(satellites)

def get_sattelites(file):
    df = pd.read_csv(file, index_col=0)
    satellites_pos = df.to_numpy()

    return satellites_pos

def renormalise_distances(features, edgeFeatures, maxDist):
    #norm = norm_file
    #df = pd.read_csv(norm, index_col=0)
    # Should only get values for imported files
    nodemin = 0
    nodemax = maxDist #df['Node dist max'].max()
    edgemin = 0 
    edgemax = maxDist #df['Edge dist max'].max()
    atommin = 0 #df['Atom min'].min()
    atommax = 95 #df['Atom max'].max()

    if nodemax < edgemax:
        nodemax = edgemax
    else:
        edgemax = nodemax

    ph = []
    for row in features:
        ph.append([renormalise(val, nodemin, nodemax) if val >= -0.5 else val for val in row])
    features = np.array(ph)
    ph = []
    for row in edgeFeatures:
        ph.append([renormalise(val, edgemin, edgemax) if val >= -0.5 else val for val in row])
    edgeFeatures = np.array(ph)

    return features, edgeFeatures

def get_data(file, path, mode):
    if mode == 'validation' or mode == 'train':
        pred_AM, label1 = read_h5py('adjPred_{:s}'.format(mode) + file, path + "/Embeddings/" )
        pred_xyz, label3 = read_h5py('satPred_{:s}'.format(mode) + file, path + "/Embeddings/")
        pred_atoms, label_atoms = read_h5py('atomPred_{:s}'.format(mode) + file, path + "/Embeddings")
    else:
        print('Wrong mode.')
        sys.exit()
    print(np.shape(pred_AM), np.shape(pred_xyz), np.shape(label1), np.shape(pred_atoms))
    return pred_AM[0], pred_xyz[0], label1[0], pred_atoms[0]


def get_files(path):
    files = os.listdir(path + '/Embeddings')
    files = sorted(np.unique([file[-14:] for file in files]))

    return files


def optimize_xyz(sat_dists, label_AM, atoms, satellites_pos, k = 10, metoden = 'L-BFGS-B', initial_guess = [0, 0, 0], debug=False):
    """

    :param sat_dists:
    :param label_AM:
    :param atom:
    :param satellites_pos:
    :param k:
    :param metoden: L-BFGS-B
    :param initial_guess:
    :return:
    """
    placed_atoms = []
    atom_list = []
    for iter, i in enumerate(sat_dists): # This code works for dummy nodes in last entry. Consider to put them in first entry instead?? Code is below!
        if np.mean(i) > -0.5:
            atom = atoms[iter].argmax().item()
            if atom != 0:
                aname = element(atom)  # This code works for dummy nodes in last entry. Consider to put them in first entry instead?? Code is below!
                atom_list.append(aname.symbol)
            if atom == 0:
                atom_list.append('D')

    atom_list = np.array(atom_list)

    for i in range(len(atom_list)):
        dist = label_AM[i]
        global history
        history = []

        inputs = []
        for j, sat in enumerate(satellites_pos):
            inputs.append(tuple((sat[0],sat[1],sat[2],sat_dists[i][j])))

        if i == 0:
            placed_atoms.append([0, 0, 0])

        else:
            #inputs = []
            for j, (dis, pos) in enumerate(zip(dist, placed_atoms)):
                inputs.append(tuple((pos[0],pos[1],pos[2],dis)))
                #print(inputs[j])

            inputs = np.array(inputs)
            results = minimize(equations,
                               initial_guess,
                               args=(inputs),
                               method=metoden,
                               #options={'initial_tr_radius':10}) #L-BFGS-B,      # The optimisation algorithm
                               options={'maxls': 100,   # Tolerance
                                        'maxiter': 1e+6,
                                        'maxfun': 1e+4,
                                        'ftol':0.0,
                                        'gtol':0.0})  # Max iterations

            data = np.array(history).T
            data = data[0]

            minidx = np.argsort(data)[:k]
            placed_atoms.append([history[minidx[0]][1], history[minidx[0]][2], history[minidx[0]][3]])

        if i != 0 and debug == True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for j, cords in enumerate(placed_atoms[-1:]):
                ax.scatter(cords[0], cords[1], cords[2], marker='o', label=atom_list[-1])

            for j, val in enumerate(inputs[4:]):
                ax.scatter(val[0], val[1], val[2], marker='.', label=atom_list[j])

                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)

                X = val[3] * np.outer(np.cos(u), np.sin(v)) + val[0]
                Y = val[3] * np.outer(np.sin(u), np.sin(v)) + val[1]
                Z = val[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + val[2]
                ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.1)

            for i, val in enumerate(inputs[:4]):
                ax.scatter(val[0], val[1], val[2], c='r', marker='.', label='Satellite {}'.format(i))

                # ax.scatter(location[0], location[1], location[2], c='r', marker='o', label='located')

                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)

                X = val[3] * np.outer(np.cos(u), np.sin(v)) + val[0]
                Y = val[3] * np.outer(np.sin(u), np.sin(v)) + val[1]
                Z = val[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + val[2]
                ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.1)

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.legend()
            plt.show()

    return atom_list, placed_atoms


if __name__ == '__main__':
    #path = './3k_files_AtomsSetToLabel'  # Which training folder should be loaded
    #mode =  'validation' # train, validation or both
    #norm_file = '/mnt/c/Users/Nanostructure/Documents/cordPred/normvaluesSatelliteDistances.csv'
    #satcsv = '/mnt/c/Users/Nanostructure/Documents/cordPred/normvaluesSatellitePositions.csv'

    files = get_files(path)
    satellites_pos = get_sattelites(satcsv)

    print('Generating XYZ files:')

    pbar = tqdm(total=len(files))
    for file in files:
        pred_AMs, pred_xyzs, label1s, pred_atoms = get_data(file, path, mode)
        for pred_AM, pred_xyz, label1, pred_atoms in zip(pred_AMs, pred_xyzs, label1s, pred_atoms):
            if '{}_{}{}_recon.xyz'.format(label1, mode, file[:-5]) in os.listdir(path + '/gen_XYZ'):
                continue
            
            pred_xyz, pred_AM = renormalise_distances(pred_xyz, pred_AM,norm_file)
            atom_list, placed_atoms = optimize_xyz(pred_xyz, pred_AM, pred_atoms, satellites_pos)
            gen_CIF(atom_list, placed_atoms, path, name='{}_{}{}_recon.xyz'.format(label1,mode, file[:-5]))

        pbar.update()
    pbar.close()
