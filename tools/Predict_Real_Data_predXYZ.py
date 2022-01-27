import pdb, time, sys, torch, os, random, glob
sys.path.append("../Data/")
from graphTools_predXYZ import movingPlot, makeData_fastload_batching
from FCC_HCP_Maker import make_structure
from nodeVAE_predXYZ import nodeVAE
import torch.optim as optim
from simPDF_xyz import *
import pandas as pd
import numpy as np
import torch.nn as nn
from manager import ml_manager
from reconstruction import get_sattelites, renormalise_distances, optimize_xyz, gen_CIF
from torch.distributions import Normal, Independent
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl  
from sklearn.manifold import TSNE
from fit import fit_PDF
np.random.seed(12)
random.seed(12)
torch.manual_seed(12)

def Reconstruct_from_PDF(epoch, model, config, PDF, atomRange, path, atom, name):
    mu, logVar, z, XYZPred, klLoc = model("random", PDF, sampling=True)  # nFeatures
    CIF_structure = []
    pred_coord = XYZPred
    for i in range(len(pred_coord)):
        if pred_coord[i][0] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][0] > config['dummy_center']-config['dummy_margin'] and pred_coord[i][1] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][1] > config['dummy_center']-config['dummy_margin'] and pred_coord[i][2] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][2] > config['dummy_center']-config['dummy_margin']:
            pass
        else:
            CIF_structure.append(pred_coord[i].clone().detach().numpy())
    CIF_structure = np.array(CIF_structure)
    atom_list = [str(atom) for i in range(len(CIF_structure))]
    gen_CIF(atom_list, CIF_structure, config['savedir'], name='{}_{}_recon.xyz'.format(epoch, name))

    cluster = loadStructure(config['savedir']+"/gen_XYZ/"+'{}_{}_recon.xyz'.format(epoch, name))
    cluster.x = cluster.x - cluster.x.mean()
    cluster.y = cluster.y - cluster.y.mean()
    cluster.z = cluster.z - cluster.z.mean()
    #cluster.write("test.xyz", format="xyz")
    dist_array = []
    for i in range(len(cluster)):
        dist = np.linalg.norm([cluster.x[i], cluster.y[i], cluster.z[i]])
        dist_array.append(dist)
    center_atom = np.argmin(dist_array)
    dist_array.sort()
    neighboor_dist = dist_array[2]

    return len(CIF_structure), neighboor_dist, z

def detect_outlier(data_1, threshold=3):
    outliers = []
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for iter, y in enumerate(data_1):
        z_score = (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(iter)
    return outliers

def remove_outliers_func(coordinate_list, threshold=3):
    outlier_structures = []
    placed_atoms = np.array(coordinate_list)
    for atom in range(np.shape(placed_atoms)[1]):
        for coordinate in range(np.shape(placed_atoms)[2]):
            outlier = detect_outlier(placed_atoms[:,atom,coordinate], threshold=threshold)
            if outlier != []:
                for number_of_outliers in range(len(outlier)):
                    outlier_structures.append(outlier[number_of_outliers])
    coordinate_list = np.delete(placed_atoms, [ np.unique(outlier_structures)], axis=0)
    return coordinate_list, len(np.unique(outlier_structures))

def Reconstruct_super_resolution(epoch, model, PDF, atomRange, atom, number_of_samples=100, remove_outliers=True, threshold=2.4, sample_graph=0, name="super_resolution"):
    coordinate_list = []
    z_list = []
    #coordinate_dict = dict()
    for i in range(number_of_samples):
        # Sampling
        mu, logVar, z, XYZPred, klLoc = model("random", PDF, sampling=True)  # nFeatures
        coordinate_list.append(XYZPred.clone().detach().numpy())
        z_list.append(z.clone().detach().numpy()[0])
    if remove_outliers:
        coordinate_list_removed_outliers, number_removed_outliers = remove_outliers_func(coordinate_list, threshold=threshold)
        print ("Removed {} outliers out of {} samples using threshold of {}".format(number_removed_outliers, number_of_samples, threshold))
        pred_coord = coordinate_list_removed_outliers
        pred_coord = torch.tensor(pred_coord).mean(0)
        CIF_structure = []
        for i in range(len(pred_coord)):
            if pred_coord[i][0] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][0] > config['dummy_center']-config['dummy_margin'] and pred_coord[i][1] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][1] > config['dummy_center']-config['dummy_margin'] and pred_coord[i][2] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][2] > config['dummy_center']-config['dummy_margin']:
                pass
            else:
                CIF_structure.append(pred_coord[i].clone().detach().numpy())  
        CIF_structure = np.array(CIF_structure)
        file_name = '{}_super_resolution_{}structures_{}_recon.xyz'.format(epoch, len(CIF_structure), name)
        gen_CIF([str(atom) for i in range(len(CIF_structure))], CIF_structure, config['savedir'], name=file_name)
    else:
        pred_coord = torch.tensor(pred_coord).mean(0)
        file_name = '{}_super_resolution_{}structures_{}_recon.xyz'.format(epoch, number_of_samples, name)
        gen_CIF([str(atom) for i in range(len(CIF_structure))], CIF_structure, config['savedir'], name=file_name)
    return z_list, len(CIF_structure)

def Traverse_LA_place(epoch, model, PDF, atomRange, atom, sample_graph=0, name="super_resolution", coordinates=[0, 0]):
    XYZ_coordinate_list = []
    number_of_atoms = []
    neighboor_dist_list = []
    # Sampling
    z = torch.tensor([coordinates], dtype=torch.float)
    XYZPreds = model.decoder(z)
    for (XYZPred, coordinate) in zip(XYZPreds, coordinates):

        CIF_structure = []
        for i in range(len(XYZPred)):
            if XYZPred[i][0] < 1.4 and XYZPred[i][0] > 0.6 and XYZPred[i][1] < 1.4 and XYZPred[i][1] > 0.6 and XYZPred[i][2] < 1.4 and XYZPred[i][2] > 0.6:
                pass
            else:
                CIF_structure.append(XYZPred[i].clone().detach().numpy())  
        CIF_structure = np.array(CIF_structure)  
        XYZ_coordinate_list.append(CIF_structure)
        number_of_atoms.append(len(CIF_structure))    
        file_name = '{}_super_resolution_{}structures_{}_recon.xyz'.format(epoch, coordinate, "specific_coordinate")
        gen_CIF([str(atom) for i in range(len(CIF_structure))], CIF_structure, config['savedir'], name=file_name)
        cluster = loadStructure(config['savedir']+"/gen_XYZ/"+file_name)
        cluster.x = cluster.x - cluster.x.mean()
        cluster.y = cluster.y - cluster.y.mean()
        cluster.z = cluster.z - cluster.z.mean()
        #cluster.write("test.xyz", format="xyz")
        dist_array = []
        for i in range(len(cluster)):
            dist = np.linalg.norm([cluster.x[i], cluster.y[i], cluster.z[i]])
            dist_array.append(dist)
        center_atom = np.argmin(dist_array)
        dist_array.sort()
        neighboor_dist = dist_array[2]
        neighboor_dist_list.append(neighboor_dist)

        generator = simPDFs_xyz()
        generator.set_parameters_xyz(rmin=0, rmax=30.1, rstep=0.01, Qmin=0.7, Qmax=22, Qdamp=0.022, Biso=0.3, delta2=0)
        generator.genPDFs_xyz(config['savedir']+"/gen_XYZ/"+file_name)
        r_constructed, Gr_constructed = generator.getPDF_xyz()
        Gr_constructed[:10] = np.zeros((10,))
        Gr_constructed /= np.max(Gr_constructed)
        plt.clf()
        plt.plot(r_constructed, Gr_constructed, label="Constructed")
        plt.savefig(config['savedir']+"/gen_XYZ/"+file_name[:-4]+".png", dpi=300)
        np.savetxt(config['savedir']+"/gen_XYZ/"+file_name[:-4]+".txt", np.column_stack([r_constructed, Gr_constructed]))

    return XYZ_coordinate_list, number_of_atoms_list, neighboor_dist_list

def make_structure_dist_from_LA(epoch, model, PDF, atomRange, atom, number_of_samples=100, sample_graph=0, name="super_resolution", coordinates=[0, 0]):
    coordinate_list = []
    number_of_atoms = []
    neighboor_dist_list = []
    #coordinate_dict = dict()
    
    z = torch.tensor([coordinates], dtype=torch.float)
    for point in range(number_of_samples):
        # Sampling   
        XYZPred = model.decoder(z)
        coordinate_list.append(XYZPred.clone().detach().numpy())
        pred_coord = XYZPred#.clone().detach().numpy()
        CIF_structure = []
        for i in range(len(pred_coord)):
            if pred_coord[i][0] < 1.4 and pred_coord[i][0] > 0.6 and pred_coord[i][1] < 1.4 and pred_coord[i][1] > 0.6 and pred_coord[i][2] < 1.4 and pred_coord[i][2] > 0.6:
                pass
            else:
                CIF_structure.append(pred_coord[i].clone().detach().numpy())  
        CIF_structure = np.array(CIF_structure)  
        number_of_atoms.append(len(CIF_structure))    
        file_name = '{}_super_resolution_{}structures_{}_{}recon.xyz'.format(epoch, coordinates, point, "specific_coordinate")
        gen_CIF([str(atom) for i in range(len(CIF_structure))], CIF_structure, config['savedir'], name=file_name)

        cluster = loadStructure(config['savedir']+"/gen_XYZ/"+file_name)
        cluster.x = cluster.x - cluster.x.mean()
        cluster.y = cluster.y - cluster.y.mean()
        cluster.z = cluster.z - cluster.z.mean()
        #cluster.write("test.xyz", format="xyz")
        dist_array = []
        for i in range(len(cluster)):
            dist = np.linalg.norm([cluster.x[i], cluster.y[i], cluster.z[i]])
            dist_array.append(dist)
        center_atom = np.argmin(dist_array)
        dist_array.sort()
        neighboor_dist = dist_array[1]
        neighboor_dist_list.append(neighboor_dist)

    return z_list

if __name__ == "__main__":
    dataDir = '../../Data_Folder'

    config = {
                #'savedir': '../../Results/All_except_HPC_200atoms_B200_HN64_LA2_XYZinput_interference_notunique',
                #'data_path': '{}/Graphs_200atoms_Qmin0p8_Qmax26_notunique'.format(dataDir),
                #'xyzPath': '{}/Graphs_200atoms_Qmin0p8_Qmax26_notunique'.format(dataDir),
                #'norm': '{}/normvaluesSatelliteDistances_100atoms_Qmin0p8_Qmax26_notunique.csv'.format(dataDir),
                #'satcsv':'{}/normvaluesSatellitePositions_0minDist_40.89780917435065maxDist_100atoms_Qmin0p8_Qmax26_notunique.csv'.format(dataDir),
                #'structure_types': ["All"],

                #'savedir': '../../Results/All_except_HPC_Deca_200atoms_B200_HN64_LA2_DefineSpace_dummycenter1_nodefinespace',
                #'data_path': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace'.format(dataDir),
                #'xyzPath': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace'.format(dataDir),
                #'norm': '{}/normvaluesSatelliteDistances_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace.csv'.format(dataDir),
                #'satcsv':'{}/normvaluesSatellitePositions_0minDist_52.84296197394692maxDist_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace.csv'.format(dataDir),
                #'structure_types': ["All"],

                'savedir': '../../Results/All_200atoms_B200_HN64_LA8_DefineSpace_dummycenter1_nodefinespace_wMoveStructures',                
                'data_path': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace_wMoveStructures'.format(dataDir),
                'xyzPath': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace_wMoveStructures'.format(dataDir),
                'norm': '{}/normvaluesSatelliteDistances_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace_wMoveStructures.csv'.format(dataDir),
                'satcsv':'{}/normvaluesSatellitePositions_0minDist_52.84296197394692maxDist_200atoms_Qmin0p8_Qmax26_ADP0p3_notunique_nodefinespace_wMoveStructures.csv'.format(dataDir),
                'structure_types': ["All"],


                'Experimental_Data': "Au144PET_100K-00000.gr",
                'Experimental_Data': "Gold144pure_110Km20007.gr",
                'atom': "Au",
                'Experimental_Data': "JQ_S3_Pt_FCC.gr",
                'atom': "Pt",
                #'Experimental_Data': "BA_PDFs_00040.gr",
                #'Experimental_Data': "BA_PDFs_00050.gr",
                #'Experimental_Data': "BB_PDFs_00045.gr",
                #'Experimental_Data': "BB_PDFs_00050.gr",
                #'Experimental_Data': "P_PDFs_00088.gr",
                #'atom': "Ir",
                
                'num_epochs' : 1,
                'nFiles' : 10,
                'B' : 10, # Set as same as nFiles
                'nhid' : 64,
                'latent_space' : 8, 
                'nNodes' : 200,
                'lr' : 5e-4,
                'satellites': 11,
                'cond_dim': 4,
                'dummy_margin': 0.4,
                'dummy_center': 1,

                'train_emb' : 10,
                'val_emb' : 10,
                'epochgrid' : 100,

                'Reconstruct_PDF': False
    }
    
    graphData, atomRange, allFiles = makeData_fastload_batching(data_path=config['data_path'], norm=config['norm'], 
        structure_type_list=config['structure_types'], batch_size=config['B'],nFiles=config['nFiles'], nNodes=config['nNodes'],
        test=True)  

    model = nodeVAE(nhid=config['nhid'], numNodesF=config['nNodes']+1+config['satellites'], satellites=config['satellites'],
        numNodes=config['nNodes'], atomRange = atomRange, latent_space=config['latent_space'], cond_dim=config['cond_dim'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0) # todo tune WD
    print("number of params: ", sum(p.numel() for p in model.parameters()))

    idx = torch.arange(len(graphData))
    vlIdx = idx[:]
    obj = ml_manager(model, config, [allFiles[idx] for idx in vlIdx], config['train_emb'], epochgrid=config['epochgrid'],
                     folderName=config['savedir'])

    obj.set_val([allFiles[idx] for idx in vlIdx], config['val_emb'])

    if obj.load == True:
        a, b = obj.continue_training()
        model.load_state_dict(torch.load(a))
        optimizer.load_state_dict(torch.load(b))
    model.eval()
    for p in model.parameters(): p.grad = None

    PDF = np.loadtxt("../../Data_Folder/Experimental_Data/"+config['Experimental_Data'], skiprows=23)

    r, Gr = PDF[:,0], PDF[:,1]
    r = np.arange(0,30.01,0.01)
    if len(Gr) < 3001:
        Gr = np.concatenate((Gr, np.zeros((3001-len(Gr)))), axis=0)
    r = r[::10]
    Gr = Gr[:3001]
    Gr = Gr[::10]
    Gr[:10] = np.zeros((10,))
    Gr /= np.max(Gr)
    plt.clf()
    plt.plot(r, Gr, label="Experimental")
    plt.plot(r, (np.array(graphData[0][6][0][:,0])), label="Simulated")
    plt.legend()
    #plt.show()
    Gr = torch.tensor(Gr, dtype=torch.float)
    Gr = Gr.unsqueeze(1).unsqueeze(0)
    name = config['Experimental_Data'][:-3]
    number_of_samples = 100
    print ("Reconstructing 1 Structure from experimental PDF")
    number_of_atoms_list = []
    neighboor_dist_list = []
    z_list = []
    
    for i in range(100):
        number_of_atoms, neighboor_dist, z = Reconstruct_from_PDF(i,  model, config, Gr, atomRange, "", config['atom'], name)
        number_of_atoms_list.append(number_of_atoms)
        neighboor_dist_list.append(neighboor_dist)
        z_list.append( z.clone().detach().numpy()[0])
    
    z_list = np.array(z_list)
    print (np.mean(z_list[:,0]), np.mean(z_list[:,1]), np.std(z_list[:,0]), np.std(z_list[:,1]))

    """
    plt.clf()
    plt.hist(number_of_atoms)
    plt.title("Number of atoms")
    plt.show()

    plt.clf()
    plt.hist(neighboor_dist_list)
    plt.title("Nearest Neighboor Distance")
    #plt.xlim(2, 3)
    plt.show()
    pdb.set_trace()
    """
    print ("Reconstructing 100 Structures from experimental PDF and make average")
    z_list, cluster_size = Reconstruct_super_resolution(obj.start_epoch, model, Gr, atomRange, atom=config['atom'], number_of_samples=number_of_samples, remove_outliers=True, threshold=2.6, sample_graph=0, name=name)
    #print (np.array(z_list)[:,0].mean(), np.array(z_list)[:,1].mean())

    """
    specific_coordinate_in_LA = [[40, -6], [30, -6], [20, -6], [10, -6], [0, -6], [-10, -6], [-20, -6], [-30, -6], [-40, -6]] 
    specific_coordinate_in_LA = []
    for percentage in np.linspace(0, 1, 201):
        x1, x2, y1, y2 = -12.18, -6.19, -20.36, -27.55 #FCC/BCC
        #x1, x2, y1, y2 = -3.5, 4.8, -45.5, -42 # BCC/SC
        #x1, x2, y1, y2 = 12, 17.5, -51, -38 # FCC/SC
        latent_point = [(percentage*x1)+(1-percentage)*x2, (percentage*y1)+(1-percentage)*y2]
        specific_coordinate_in_LA.append(latent_point)
    XYZ_coordinate_list, number_of_atoms_list, neighboor_dist_list = Traverse_LA_place(obj.start_epoch, model, Gr, atomRange, atom=config['atom'], sample_graph=0, name=name, coordinates=specific_coordinate_in_LA)
    #z_list = make_structure_dist_from_LA(obj.start_epoch, model, Gr, atomRange, atom=config['atom'], number_of_samples=number_of_samples, sample_graph=0, name=name, coordinates=specific_coordinate_in_LA)
    """
    
    generator = simPDFs_xyz()
    generator.set_parameters_xyz(rmin=0, rmax=30.1, rstep=0.1, Qmin=0.7, Qmax=22, Qdamp=0.022, Biso=0.3, delta2=0)
    file_name = '{}_super_resolution_{}structures_{}_recon.xyz'.format(obj.start_epoch, cluster_size, name)
    #file_name = '84_JQ_S3_Pt_FCC_recon.xyz'
    #file_name = '9_Os_200mM_recon.xyz'
    #file_name = "8_Insitu_PdIn_BW_1-0050_recon.xyz"
    #file_name = "48_Insitu_PdIn_BW_1-0010_recon.xyz"

    generator.genPDFs_xyz(config['savedir']+"/gen_XYZ/"+file_name)
    r_constructed, Gr_constructed = generator.getPDF_xyz()
    Gr_constructed[:10] = np.zeros((10,))
    Gr_constructed /= np.max(Gr_constructed)
    plt.clf()
    plt.plot(r,np.array(Gr[0,:,0]), label="Experimental")
    plt.plot(r_constructed, Gr_constructed, label="Constructed")
    plt.legend()
    plt.show()

    new_PDF = "../../Data_Folder/Experimental_Data/"+config['Experimental_Data'][:-3]+"_normalised.gr"
    np.savetxt(new_PDF, np.column_stack([r, np.array(Gr[0,:,0])]))
    r, g, gcalc, diff, diffzero = fit_PDF(new_PDF, Qmin=0.7, Qmax=20, Qdamp=0.03, XYZ_file=config['savedir']+"/gen_XYZ/"+file_name)
    np.savetxt(config['savedir']+"/gen_XYZ/"+file_name[:-4]+"fit.txt", np.column_stack([r, g, gcalc, diff, diffzero]))

