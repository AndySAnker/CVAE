import os, time, multiprocessing, math, random, torch, h5py, glob
from simPDF_xyz import *
from mendeleev import element
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from shutil import copyfile
from diffpy.srreal.overlapcalculator import OverlapCalculator
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from diffpy.Structure import loadStructure
from norm_space import define_space
from sampleSphere import sampleSphere
import matplotlib.pyplot as plt
from plot_structure import plot_structure
torch.manual_seed(12)
random.seed(12)
np.random.seed(12)

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

def getMinMaxVals(norm, nFiles, maxDist):
    print(norm)
    df = pd.read_csv(norm)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    files = []
    atomMin = 99
    atomMax = 0
    minDist = 0
    maxDist = maxDist # Largest distance in all structures
    for index, row in df.iterrows():
        if index == nFiles:
            break
        namePh = row[row.index[0]]
        files.append(namePh[:-4])

        if atomMin > row['Atom min']:
            atomMin = row['Atom min']

        if row['Atom max'] > atomMax:
            atomMax = row['Atom max']

        #if minDist > row['Edge dist min']:
        #    minDist = np.amin([row['Edge dist min']])

        #if row['Edge dist max'] > maxDist:
        #    maxDist = np.amax(np.amax([row['Edge dist max']]))

    return atomMin, atomMax, minDist, maxDist, files

def transformers(edges,eFeatures,nNodes, normMin, normMax, deadVal):
    """ Obtain source and sink node transformer matrices"""
    softAdj = torch.sparse.FloatTensor(edges,eFeatures,(nNodes,nNodes))
    softAdj = softAdj.to_dense()
    for i in range(nNodes):
        for j in range(i,nNodes):
            if softAdj[i][j] != 0.:
                softAdj[i][j] = normFeatures(softAdj[i][j], normMin, normMax)
            else:
                if i == j and softAdj[0][j] != deadVal:
                    softAdj[i][j] = 0
                elif i == j:
                    softAdj[i][j] = 0.5*deadVal  # Diagonal is added to itself
                else:
                    softAdj[i][j] = deadVal

    softAdj += softAdj.T

    return softAdj

def makeData(data_path, xyzPath, norm, nNodes=100, nFiles=100, loadPDF=False, test_set_ratio=0.2, deadVal=-1, minDist=0 ,maxDist=100):
    start_time = time.time()
    """
    Input: Directory with .txt files
    Output: List of graphs with node features and adjacency
    """
    #atomMin, atomMax, minDist, maxDist, files = getMinMaxVals(norm, nFiles, maxDist)
    if not os.path.isdir(data_path + "/fastload"):
        os.mkdir(data_path + "/fastload")
    if not os.path.isdir(data_path + "/test_set"):
        os.mkdir(data_path + "/test_set")
    destination = "/fastload/"
    
    files = sorted(glob.glob(data_path+'*h5*'))

    atomMin = 0
    atomMax = 95
    atomRange = (atomMax - atomMin) + 1
    N = len(files)
    random.shuffle(files)
    print('Found %d graph files' % (N))
    graphData = []

    pbar = tqdm(total=nFiles)
    for iter, f in enumerate(files):
        f = f.replace(data_path, "")[:-3]
        if not os.path.isfile(data_path + destination[1:] + f + '.h5'):
            loadData = h5py.File(data_path + f + '.h5', 'r')
            nFeatures = loadData.get('NFM')
            eFeatures = loadData.get('EFM')
            edgeIndex = loadData.get('EAR')
        
            edgeIndex = torch.LongTensor(np.array(edgeIndex, dtype=int))
            eFeatures = torch.Tensor(eFeatures[:,0])
            nFeatures = torch.FloatTensor(nFeatures)
            
	        ### Cast to Torch
            edgeIndex = torch.LongTensor(np.array(edgeIndex, dtype=int))
            eFeatures = torch.Tensor(eFeatures)
            nFeatures = torch.FloatTensor(nFeatures)
	        
	        ### Obtain the source and sink node pairs
            softAdj = transformers(edgeIndex, eFeatures, nNodes, minDist, maxDist, deadVal)

            ph = nFeatures.T.clone()

            atomLabels = torch.zeros((nNodes, atomRange))  # Last index is for dummy nodes
            for j, val in enumerate(ph[0].T):
                if val != deadVal:  # If not dummy node
                    atomLabels[j][int(val)] = 1
                else:  # If dummy node
                    atomLabels[j][0] = 1
            atoms = torch.tensor([normFeatures(val, atomMin, atomMax) if val != deadVal else val for val in ph[0]]).unsqueeze(1)

            y_atom = torch.zeros((atomRange, 1))
            for i in range(len(ph[0])):
                if int(ph[0][i]) != deadVal:
                    y_atom[int(ph[0][i])] = 1
                else:
                    y_atom[0] = 1

            for i in range(ph[1:].size()[0]):
                for j in range(ph[1:].size()[1]):
                    if ph[1:][i][j] == deadVal:
                        break
                    else:
                        ph[1:][i][j] = normFeatures(ph[1:][i][j], minDist, maxDist)
	        
            satLabels = ph[1:].T
            nFeatures = torch.cat((atoms, satLabels, softAdj), dim=1)
            if loadPDF == False:
                generator = simPDFs_xyz()
                generator.set_parameters_xyz(rmin=0, rmax=30.1, rstep=0.1, Qmin=0.8, Qmax=26, Qdamp=0.03, Biso=0.3, delta2=0)
                generator.genPDFs_xyz(xyzPath+'/{}.xyz'.format(f))
                r, Gr = generator.getPDF_xyz()
            else:
                PDF = np.loadtxt(data_path + '/' + f + '_PDF.txt', delimiter=' ')
                Gr = PDF[:, 1]
            Gr[:10] = torch.zeros((10,))
            #Gr += np.abs(np.min(Gr))
            Gr /= np.max(Gr)
            Gr = torch.tensor(Gr, dtype=(torch.float)).view(1, len(Gr), 1)

            if not iter < (1 - test_set_ratio) * len(files):
                destination = "/test_set/"

            save_graphData = dict(hf_atomLabels=atomLabels, hf_nFeatures=nFeatures, hf_satLabels=satLabels, hf_softAdj=softAdj, hf_y_atom=y_atom, hf_Gr=Gr)
            hf_graphData = h5py.File(data_path + destination + f + '.h5', 'w')
            for dict_label, dict_Data in save_graphData.items():
                hf_graphData.create_dataset(dict_label, data=dict_Data)
            hf_graphData.close()
            
            graphData.append((f, atomLabels, nFeatures, satLabels, softAdj, y_atom, Gr))
        pbar.update(1)
    print("Time used to load data:", (time.time() - start_time) / 60, "min")
    pbar.close()
    return graphData, atomRange, files

def create_graphs(files, label, saveFolder, satellites, nAtoms, return_dict):
    renorm_dict = {}
    bc = OverlapCalculator()
    bc.atomradiitable = 'covalent'
    pbar = tqdm(total=len(files))
    for file in files:
        if os.path.isfile(saveFolder +'/'+ str(file[0:-4]) + ".h5"):
            print (file , "Already Made")
        else:
            print(file)
            cluster, stru_dim = define_space(file, saveFolder, nAtoms, plot=False)
            #cluster = loadStructure(saveFolder +'/'+ file)
            stru_dim = "3D"

            NFM = np.zeros((numb_nodes,numb_nodeF)) + deadVal
            ele_min = 99
            ele_max = 0
            for i, cords in enumerate(cluster):

                atompos = np.array([cluster[i].x, cluster[i].y, cluster[i].z])
                ele_sym = cluster.element[i]
                atomnumber = element(ele_sym).atomic_number

                if atomnumber > ele_max:
                    ele_max = atomnumber
                if ele_min > atomnumber:
                    ele_min = atomnumber

                NFM[i][0] = atomnumber
                for j, sat in enumerate(satellites):
                    NFM[i][j+1] = np.linalg.norm(atompos - sat)

            NFM_graphData = NFM.copy()

            NFM = NFM.T
            NFM = NFM[1:]
            dist_min = np.min(NFM[NFM != deadVal])
            dist_max = np.max(NFM[NFM != deadVal])


            # Edge features
            G = nx.Graph()  # create an empty graph with no nodes and no edges
            G.add_nodes_from(range(numb_nodes))  # add a list of nodes

            dist_list = []
            index = []
            for i in range(len(cluster)):
                atom_neighbors_1 = bc.getNeighborSites(i)

                ph_index = []
                for j in range(i+1,len(cluster)):
                    position_0 = cluster.x[i], cluster.y[i], cluster.z[i]
                    position_1 = cluster.x[j], cluster.y[j], cluster.z[j]
                    new_dist = distance(position_0, position_1)
                    dist_list.append(new_dist)
                    ph_index.append([i,j])
                    G.add_edges_from([(i, j, {'Distance': new_dist})])

                index.append(ph_index)

            dist_list = np.array(dist_list)

            dist_min2 = np.min(dist_list)
            dist_max2 = np.max(dist_list)


            Edge_feature_matrix = np.zeros([1, len(G.edges)])  # This should be dynamic

            for count, edge in enumerate(G.edges):
                Edge_feature_matrix[:][0][count] = nx.get_edge_attributes(G, "Distance")[edge]

            edge_list = []
            for edge in G.edges:
                edge_list.append([edge[0], edge[1]])

            Edge_array = np.asarray(edge_list).T

            graphData = dict(NFM=NFM_graphData, EFM=Edge_feature_matrix.T, EAR=Edge_array)
            hf_graphData = h5py.File(saveFolder +'/'+ str(file[0:-4]) + '.h5', 'w')
            for dict_label, dict_Data in graphData.items():
                hf_graphData.create_dataset(dict_label, data=dict_Data)
            hf_graphData.close()
        

            renorm_dict.update({'{}'.format(file) : {'Node dist min' : dist_min,'Node dist max' : dist_max,
                                                     'Edge dist min' : dist_min2, 'Edge dist max' : dist_max2,
                                                     'Atom min' : ele_min, 'Atom max' : ele_max,
                                                     'Structure dim' : stru_dim}})

        pbar.update(1)

    pbar.close()

    df = pd.DataFrame.from_dict(renorm_dict, orient="index")
    return_dict[label] = df
    return None

def normFeatures(x, min, max):
    return (x-min)/(max-min)


def distance(position_0, position_1):
    """ Returns the distance between vectors position_0 and position_1 """
    return np.sqrt((position_0[0] - position_1[0]) ** 2 + (position_0[1] - position_1[1]) ** 2 + (
                position_0[2] - position_1[2]) ** 2)


def dist_check(cluster, bc, overlap = 0.7):
    for idx1 in range(len(cluster)):
        for idx2 in range(idx1 + 1, len(cluster)):
            pos1 = np.array([cluster[idx1].x, cluster[idx1].y, cluster[idx1].z])
            pos2 = np.array([cluster[idx2].x, cluster[idx2].y, cluster[idx2].z])
            dist1 = distance(pos1, pos2)
            dist2 = bc.atomradiitable.lookup(cluster.element[idx1]) + bc.atomradiitable.lookup(cluster.element[idx2])
            if dist2 * overlap > dist1:
                return False
    return True

def getMinMaxDist(cluster):
    minDist = 99
    maxDist = 0
    for idx1 in range(len(cluster)):
        for idx2 in range(idx1 + 1, len(cluster)):
            pos1 = np.array([cluster[idx1].x, cluster[idx1].y, cluster[idx1].z])
            pos2 = np.array([cluster[idx2].x, cluster[idx2].y, cluster[idx2].z])
            dist1 = distance(pos1, pos2)

            if dist1 > maxDist:
                maxDist = dist1
            elif minDist > dist1:
                minDist = dist1

    return minDist, maxDist

def structure_check(files, folder, folder_save):
    ph = files.copy()
    bc = OverlapCalculator()
    bc.atomradiitable = 'covalent'
    distance_list = []
    fails = []
    print('\nGetting min and max for atomnumber:')
    distList = []
    pbar = tqdm(total=len(files))
    for count, file in enumerate(files):
        try:
            cluster = loadStructure(folder + '/' + file)
            dpc = DebyePDFCalculator()
            _, _ = dpc(cluster)
            bc(cluster)
        except:
            ph.remove(file)
            fails.append(file)
            pbar.update(1)
            continue

        if len(cluster) < minimum_atoms or len(cluster) > maximum_atoms:

            ph.remove(file)
            pbar.update(1)
            continue

        minDist, maxDist = getMinMaxDist(cluster)
        distList.append([minDist, maxDist])
        if not os.path.isfile(folder_save+'/'+file):
            copyfile(folder+'/'+file,folder_save+'/'+file)
        pbar.update(1)

    pbar.close()
    distList = np.array(distList)
    maxmax = np.amax(distList)
    minmin = np.amin(distList)

    if fails != []:
        print('Following files failed loading:')
        for file in fails:
            print(file)
    else:
        print('No files failed loading')

    print ("MinDist found to: ", minmin)
    print ("MaxDist found to: ", maxmax)
    return ph, minmin, maxmax

def move_structures(enliste, folder):
    for file in enliste:
        cluster = loadStructure(folder + '/' + file)
        #if cluster.x.mean() != 0 or cluster.y.mean() != 0 or cluster.z.mean() != 0:
        cluster.x = cluster.x - cluster.x.mean()
        cluster.y = cluster.y - cluster.y.mean()
        cluster.z = cluster.z - cluster.z.mean()
        cluster.write(folder + '/' + file, format="xyz")
    return

if __name__ == '__main__':
    global root  # Root folder for script
    global folder  # Folder where xyz are fetched
    global saveFolder  # Where the graphs will be saved
    global deadVal  # Value assigned dummy nodes
    global numb_nodes  # Maximum number of nodes in graph, numb_nodes = maximum_atoms
    global numb_nodeF  # Node features: x, y, z and atom number
    global minimum_atoms  # Smallest structural motif
    global maximum_atoms  # Largest structural motif, numb_nodes = maximum_atoms

    numb_lists = 8  # The number of threads generated
    numSats = 11
    satellites = sampleSphere(numSats)
    index = ['Satellite{}'.format(i) for i in range(np.shape(satellites)[0])]

    # Define values
    deadVal = -1
    numb_nodes = 200
    numb_nodeF = 1+np.shape(satellites)[0]
    minimum_atoms = 4
    maximum_atoms = numb_nodes

    # Placeholders
    atom_min = 99
    atom_max = 0

    # Get files
    root = os.getcwd()
    folder = '/mnt/c/Users/Nanostructure/Downloads/CVAE/makeData/XYZ_200atoms/'
    enliste = sorted(os.listdir(folder))
    enliste = [file for file in enliste if file[0] != '.' and file[-4:] == '.xyz']
    saveFolder = '/mnt/c/Users/Nanostructure/Downloads/CVAE/makeData/Graphs_200atoms/' 
    satPath = '/mnt/c/Users/Nanostructure/Downloads/CVAE/makeData/'
    shall_we_check_structures = True
    np.random.shuffle(enliste)
    #enliste = enliste[:10]

    if shall_we_check_structures:
        enliste, minDist, maxDist = structure_check(enliste, folder, saveFolder)
        minDist = 0
        move_structures(enliste, folder) # Move middle atom_1transs into origo
    satellites *= 0.5*maxDist # Half of the longest distance
    start_time = time.time()
    
    print(np.shape(satellites))
    df = pd.DataFrame(satellites, index=[index],
                                  columns=['x','y','z'])
    df.to_csv(satPath+'/normvaluesSatellitePositions_{}minDist_{}maxDist_200atoms_Qmin0p8_Qmax26_ADP0p3.csv'.format(minDist, maxDist))
    print('\n{} xyz files matched search criteria'.format(len(enliste)))

    idx = math.ceil(len(enliste)/numb_lists)
    list_of_slices = []
    for i in range(numb_lists):
        list_of_slices.append(enliste[i*idx:(1+i)*idx])
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for label, i in enumerate(list_of_slices):
        p = multiprocessing.Process(target=create_graphs, args=(i, label, saveFolder, satellites, numb_nodes, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print('\nCreating renorm csv')
    for i, dicts in enumerate(return_dict.values()):
        if i == 0:
            df = dicts
        else:
            df = pd.concat([dicts,df])
    df.to_csv(satPath+'/normvaluesSatelliteDistances_200atoms_Qmin0p8_Qmax26_ADP0p3.csv')

    print('Took {:.2f} min'.format((time.time() - start_time)/60))
    
    print ("Normalising all the data and save it as h5py ready to fastload and splitting the data into training/test set.")
    graphData, atomRange, allFiles = makeData(data_path=saveFolder,
                                              xyzPath=saveFolder,
                                              norm=satPath+'/normvaluesSatelliteDistances_200atoms_Qmin0p8_Qmax26_ADP0p3.csv',
                                              nFiles=len(enliste),
                                              nNodes=numb_nodes,
                                              loadPDF=False, # Either generates or reads PDFs
                                              test_set_ratio = 0.2,
                                              deadVal = deadVal,
                                              maxDist = maxDist)  


