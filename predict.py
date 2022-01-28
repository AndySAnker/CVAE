import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tools.graphTools import makeData_fastload_batching
from tools.nodeVAE import nodeVAE
from tools.manager import ml_manager
from tools.reconstruction import gen_CIF
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

    return len(CIF_structure), z

if __name__ == "__main__":
    dataDir = './dataset/'

    config = {
                'savedir': 'modelsANDresults/All_200atoms_B200_HN64_LA8',                
                'data_path': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3'.format(dataDir),
                'xyzPath': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3'.format(dataDir),
                'norm': '{}/normvaluesSatelliteDistances_200atoms_Qmin0p8_Qmax26_ADP0p3.csv'.format(dataDir),
                'satcsv':'{}/normvaluesSatellitePositions_0minDist_52.84296197394692maxDist_200atoms_Qmin0p8_Qmax26_ADP0p3.csv'.format(dataDir),
                'structure_types': ["All"],

                'Experimental_Data': "S3_Pt.gr",
                'atom': "Pt",
                
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
    }
    
    # Load data not in graph format but inspired by graph-data
    graphData, atomRange, allFiles = makeData_fastload_batching(data_path=config['data_path'], norm=config['norm'], 
        structure_type_list=config['structure_types'], batch_size=config['B'],nFiles=config['nFiles'], nNodes=config['nNodes'],
        test=True)  

    # Setup model architecture
    model = nodeVAE(nhid=config['nhid'], numNodesF=config['nNodes']+1+config['satellites'], satellites=config['satellites'],
        numNodes=config['nNodes'], atomRange = atomRange, latent_space=config['latent_space'], cond_dim=config['cond_dim'])

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0) # todo tune WD
    print("number of params: ", sum(p.numel() for p in model.parameters()))

    # Use manager tool to follow and save training
    idx = torch.arange(len(graphData))
    vlIdx = idx[:]
    obj = ml_manager(model, config, [allFiles[idx] for idx in vlIdx], config['train_emb'], epochgrid=config['epochgrid'],
                     folderName=config['savedir'])

    obj.set_val([allFiles[idx] for idx in vlIdx], config['val_emb'])

    # Load model
    if obj.load == True:
        a, b = obj.continue_training()
        model.load_state_dict(torch.load(a))
        optimizer.load_state_dict(torch.load(b))
    model.eval()
    for p in model.parameters(): p.grad = None

    # Load experimental PDF
    PDF = np.loadtxt("exp_data/"+config['Experimental_Data'], skiprows=23) # Set up to load PDFGui files

    # Make PDF to r-steps of 0.1 Å and to be between 0 - 30 Å. Also normalise to have highest peak to 1.
    r, Gr = PDF[:,0], PDF[:,1]
    plt.plot(r, Gr, label="Experimental - before preprocessing")
    r = np.arange(0,30.01,0.1)
    if len(Gr) < 3001:
        Gr = np.concatenate((Gr, np.zeros((3001-len(Gr)))), axis=0) # Pad PDF if less than data up to 30 Å
    Gr = Gr[:3001] # Keep data up to 30 Å and remove rest
    Gr = Gr[::10] # Only use every 10th point. Is like nyquist sampling
    Gr[:10] = np.zeros((10,)) # Do not use the data up to 1 Å
    Gr /= np.max(Gr) # Normalise the data
    plt.plot(r, Gr, label="Experimental - after preprocessing")
    plt.legend()
    plt.show()
    # Make PDF ready to use for prediction
    Gr = torch.tensor(Gr, dtype=torch.float)
    Gr = Gr.unsqueeze(1).unsqueeze(0)
    name = config['Experimental_Data'][:-3]
    number_of_samples = 100 # How many samples should be drawn from latent space
    print ("Reconstructing 1 Structure from experimental PDF")
    number_of_atoms_list = []
    z_list = []
    
    # Predict on PDF using the CVAE
    for i in range(100):
        number_of_atoms, z = Reconstruct_from_PDF(i,  model, config, Gr, atomRange, "", config['atom'], name)
        number_of_atoms_list.append(number_of_atoms)
        z_list.append( z.clone().detach().numpy()[0])
    
    z_list = np.array(z_list)
    print (np.mean(z_list[:,0]), np.mean(z_list[:,1]), np.std(z_list[:,0]), np.std(z_list[:,1]))

