import glob, random, time, torch, h5py, pdb, mendeleev
#from make_XYZ_files import find_structure_type
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
torch.manual_seed(12)
random.seed(12)

def makeData_fastload_batching(data_path, norm, batch_size, structure_type_list=["ALL"], nNodes=100, nFiles=100, test=False, dummy_center=1):
    start_time = time.time()
    """
    Input: Directory with .txt files
    Output: List of graphs with node features and adjacency
    """
    destination = "/fastload/"
    if test:
        destination = "/test_set/"

    atomRange = 96
    if structure_type_list == ["All"]:
        files = sorted(glob.glob(data_path+destination+'*.h5*'))
    else:
        for iter, structure_type in enumerate(structure_type_list):
            files_ph = sorted(glob.glob(data_path+destination+'{}*.h5*'.format(structure_type)))
            if iter == 0:
                files = files_ph
            else:
                files = files + files_ph
    random.shuffle(files)
    files = files[:nFiles]
    print('Found %d graph files' % (len(files)))
    graphData = []
    f_list = []

    pbar = tqdm(total=len(files))
    for iter, f in enumerate(files):
        f = f.replace(data_path+destination[:-1], "")[1:-3]
        f_list.append(f)

        loadData = h5py.File(data_path+destination+f+'.h5', 'r')
        atomLabels = loadData.get('hf_atomLabels')
        nFeatures = loadData.get('hf_nFeatures')
        satLabels = loadData.get('hf_satLabels')
        softAdj = loadData.get('hf_softAdj')
        y_atom = loadData.get('hf_y_atom')
        Gr = loadData.get('hf_Gr')
        df_XYZ_true = pd.read_csv(data_path+"/"+f+".xyz", skiprows=2, sep=' ', skipinitialspace=1, names=["atom", "x", "y", "z"]) 
        true_coord = torch.tensor(df_XYZ_true[["x","y","z"]].values, dtype=torch.float)
        true_coord = torch.cat((true_coord, dummy_center*torch.ones((nNodes-len(true_coord), 3))), dim=0).unsqueeze(0)

        if batch_size == 1:
            atomLabels_ph = torch.tensor(atomLabels, dtype=torch.float).unsqueeze(0)
            nFeatures_ph = torch.tensor(nFeatures, dtype=torch.float).unsqueeze(0)
            satLabels_ph = torch.tensor(satLabels, dtype=torch.float).unsqueeze(0)
            softAdj_ph = torch.tensor(softAdj, dtype=torch.float).unsqueeze(0)
            y_atom_ph = torch.tensor(y_atom, dtype=torch.float).unsqueeze(0)
            Gr = torch.tensor(Gr, dtype=(torch.float))
            Gr_ph = Gr.view(1, Gr.size()[1],1).unsqueeze(0)
            structure_type_ph = torch.tensor(structure_type, dtype=torch.float).unsqueeze(0)
            graphData.append((f_list, atomLabels_ph, nFeatures_ph, satLabels_ph, softAdj_ph, y_atom_ph, Gr_ph.squeeze(1)))
            break
        if (iter+1) % batch_size == 1:
            atomLabels_ph = torch.tensor(atomLabels, dtype=torch.float).unsqueeze(0)
            nFeatures_ph = torch.tensor(nFeatures, dtype=torch.float).unsqueeze(0)
            satLabels_ph = torch.tensor(satLabels, dtype=torch.float).unsqueeze(0)
            softAdj_ph = torch.tensor(softAdj, dtype=torch.float).unsqueeze(0)
            y_atom_ph = torch.tensor(y_atom, dtype=torch.float).unsqueeze(0)
            Gr = torch.tensor(Gr, dtype=(torch.float))
            Gr_ph = Gr.view(1, Gr.size()[1],1).unsqueeze(0)
            true_coord_ph = true_coord
        else:
            atomLabels_ph = torch.cat((atomLabels_ph, torch.tensor(atomLabels, dtype=torch.float).unsqueeze(0)), dim=0)
            nFeatures_ph = torch.cat((nFeatures_ph, torch.tensor(nFeatures, dtype=torch.float).unsqueeze(0)), dim=0)
            satLabels_ph = torch.cat((satLabels_ph, torch.tensor(satLabels, dtype=torch.float).unsqueeze(0)), dim=0)
            softAdj_ph = torch.cat((softAdj_ph, torch.tensor(softAdj, dtype=torch.float).unsqueeze(0)), dim=0)
            y_atom_ph = torch.cat((y_atom_ph, torch.tensor(y_atom, dtype=torch.float).unsqueeze(0)), dim=0)
            Gr = torch.tensor(Gr, dtype=(torch.float))
            Gr_ph = torch.cat((Gr_ph, Gr.view(1, Gr.size()[1],1).unsqueeze(0)), dim=0)
            true_coord_ph = torch.cat((true_coord_ph, true_coord), dim=0)

        if (iter+1) % batch_size == 0:
            nFeatures_ph = torch.cat((nFeatures_ph, true_coord_ph), dim=2)
            graphData.append((f_list, atomLabels_ph, nFeatures_ph, satLabels_ph, softAdj_ph, y_atom_ph, Gr_ph.squeeze(1)))
            f_list = []    
        pbar.update(1)    
    print ("Time used to load data:", (time.time()-start_time)/60, "min")
    pbar.close()
    return graphData, atomRange, files

def makeData_fastload_batching_rotate(data_path, norm, batch_size, nNodes=100, nFiles=100, test=False):
    start_time = time.time()
    """
    Input: Directory with .txt files
    Output: List of graphs with node features and adjacency
    """
    destination = "/fastload/"
    if test:
        destination = "/test_set/"

    atomRange = 96
    files = sorted(glob.glob(data_path+destination+'*.h5*'))
    random.shuffle(files)
    files = files[:nFiles]
    print('Found %d graph files' % (len(files)))
    graphData = []
    f_list = []

    pbar = tqdm(total=len(files))
    for iter, f in enumerate(files):
        f = f.replace(data_path+destination[:-1], "")[1:-3]
        f_list.append(f)

        loadData = h5py.File(data_path+destination+f+'.h5', 'r')
        atomLabels = loadData.get('hf_atomLabels')
        nFeatures = loadData.get('hf_nFeatures')
        satLabels = loadData.get('hf_satLabels')
        softAdj = loadData.get('hf_softAdj')
        y_atom = loadData.get('hf_y_atom')
        Gr = loadData.get('hf_Gr')

        if batch_size == 1:
            atomLabels_ph = torch.tensor(atomLabels, dtype=torch.float).unsqueeze(0)
            nFeatures_ph = torch.tensor(nFeatures, dtype=torch.float).unsqueeze(0)
            satLabels_ph = torch.tensor(satLabels, dtype=torch.float).unsqueeze(0)
            softAdj_ph = torch.tensor(softAdj, dtype=torch.float).unsqueeze(0)
            y_atom_ph = torch.tensor(y_atom, dtype=torch.float).unsqueeze(0)
            Gr = torch.tensor(Gr, dtype=(torch.float))
            Gr_ph = Gr.view(1, Gr.size()[1],1).unsqueeze(0)
            structure_type_ph = torch.tensor(structure_type, dtype=torch.float).unsqueeze(0)
            graphData.append((f_list, atomLabels_ph, nFeatures_ph, satLabels_ph, softAdj_ph, y_atom_ph, Gr_ph.squeeze(1)))
            break
        if (iter+1) % batch_size == 1:
            atomLabels_ph = torch.tensor(atomLabels, dtype=torch.float).unsqueeze(0)
            nFeatures_ph = torch.tensor(nFeatures, dtype=torch.float).unsqueeze(0)
            satLabels_ph = torch.tensor(satLabels, dtype=torch.float).unsqueeze(0)
            softAdj_ph = torch.tensor(softAdj, dtype=torch.float).unsqueeze(0)
            y_atom_ph = torch.tensor(y_atom, dtype=torch.float).unsqueeze(0)
            Gr = torch.tensor(Gr, dtype=(torch.float))
            Gr_ph = Gr.view(1, Gr.size()[1],1).unsqueeze(0)
        else:
            atomLabels_ph = torch.cat((atomLabels_ph, torch.tensor(atomLabels, dtype=torch.float).unsqueeze(0)), dim=0)
            nFeatures_ph = torch.cat((nFeatures_ph, torch.tensor(nFeatures, dtype=torch.float).unsqueeze(0)), dim=0)
            satLabels_ph = torch.cat((satLabels_ph, torch.tensor(satLabels, dtype=torch.float).unsqueeze(0)), dim=0)
            softAdj_ph = torch.cat((softAdj_ph, torch.tensor(softAdj, dtype=torch.float).unsqueeze(0)), dim=0)
            y_atom_ph = torch.cat((y_atom_ph, torch.tensor(y_atom, dtype=torch.float).unsqueeze(0)), dim=0)
            Gr = torch.tensor(Gr, dtype=(torch.float))
            Gr_ph = torch.cat((Gr_ph, Gr.view(1, Gr.size()[1],1).unsqueeze(0)), dim=0)

        if (iter+1) % batch_size == 0:
            graphData.append((f_list, atomLabels_ph, nFeatures_ph, satLabels_ph, softAdj_ph, y_atom_ph, Gr_ph.squeeze(1)))
            f_list = []    
        pbar.update(1)    
    print ("Time used to load data:", (time.time()-start_time)/60, "min")
    pbar.close()
    return graphData, atomRange, files

def movingPlot(data, name, epochs, plotTitle,box):
    matplotlib.use('Agg') # Allow saving of figure without program X opened
    plt.style.use('ggplot')
    colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    LINE_SIZE = 1.5
    MARKER_SIZE = 10
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 35
    FIGSIZE = (16,8)

    matplotlib.rcParams['font.family'] = "sans-serif"
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE + 3)  # fontsize of the tick labels
    plt.rc('xtick.major', width=LINE_SIZE, size=LINE_SIZE+ 3)  # x ticks and size
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick.major', width=LINE_SIZE, size=LINE_SIZE+ 3)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('figure', figsize=FIGSIZE)
    plt.rcParams['lines.linewidth'] = LINE_SIZE  # fontsize of the figure title
    plt.rcParams['lines.markersize'] = MARKER_SIZE  # fontsize of the figure title
    x = range((epochs+1)-box,epochs+1)
    plt.hlines(data[0][-box],x[0], x[-1], linestyles='--')
    plt.hlines(data[1][-box],x[0], x[-1])

    plt.plot(x, data[0][-box:],'--',label='{}'.format(name[0]))
    plt.plot(x, data[1][-box:], label='{}'.format(name[1]))

    plt.legend()

    plt.ylabel('{}'.format(plotTitle))
    plt.xlabel('Epoch')

    plt.title('{}'.format(plotTitle))
    plt.legend()
    plt.tight_layout()
    plt.savefig('./img/{}_{:05d}_{:05d}.png'.format(plotTitle,epochs-box,epochs), dpi=300)
    plt.clf()

    return None
