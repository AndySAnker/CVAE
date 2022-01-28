import time, torch, datetime, sys
import numpy as np
import pandas as pd
import torch.optim as optim
from tools.graphTools import movingPlot, makeData_fastload_batching
from tools.nodeVAE import nodeVAE, reconstruction_loss
from tools.manager import ml_manager
from tools.reconstruction import gen_CIF

def make_CIF(structures, XYZPreds, config, epoch, atomRange = 96, inference=True):
    Atoms_off = []
    for iter, (structure, pred_coord) in enumerate(zip(structures, XYZPreds)):
        df_XYZ_true = pd.read_csv(config['data_path']+"/"+structure+".xyz", skiprows=2, sep=" ", names=["atom", "x", "y", "z"])
        true_coord = torch.tensor(df_XYZ_true[["x","y","z"]].values, dtype=torch.float) 
        CIF_structure = []
        for i in range(len(pred_coord)):
            if pred_coord[i][0] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][0] > config['dummy_center']-config['dummy_margin'] and pred_coord[i][1] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][1] > config['dummy_center']-config['dummy_margin'] and pred_coord[i][2] < config['dummy_center']+config['dummy_margin'] and pred_coord[i][2] > config['dummy_center']-config['dummy_margin']:
                pass
            else:
                CIF_structure.append(pred_coord[i].clone().detach().numpy())
        CIF_structure = np.array(CIF_structure)
        if inference:
            gen_CIF(["Ag" for i in range(len(CIF_structure))], CIF_structure, config['savedir'], name='valset_inference_{}_{}_{}_recon.xyz'.format(epoch, 'sampling', structure))
        else:
            gen_CIF(["Ag" for i in range(len(CIF_structure))], CIF_structure, config['savedir'], name='valset_{}_{}_{}_recon.xyz'.format(epoch, 'sampling', structure))
        Atoms_off.append(len(true_coord) - len(CIF_structure))     
    return Atoms_off 

torch.manual_seed(12)
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
use_cuda = False
dataDir = './dataset/'

config = {
            'savedir': 'modelsANDresults1/All_200atoms_B200_HN64_LA8',

            'data_path': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3'.format(dataDir),
            'xyzPath': '{}/Graphs_200atoms_Qmin0p8_Qmax26_ADP0p3'.format(dataDir),
            'norm': '{}/normvaluesSatelliteDistances_200atoms_Qmin0p8_Qmax26_ADP0p3.csv'.format(dataDir),
            'satcsv':'{}/normvaluesSatellitePositions_0minDist_52.84296197394692maxDist_200atoms_Qmin0p8_Qmax26_ADP0p3.csv'.format(dataDir),
            'structure_types': ["All"],

            'num_epochs' : 100000,
            'nFiles' : 2932, #5680, #FCC: 96  HCP: 1767  All: 2932
            'beta' : 1,
            'Adj_loss_factor': 1,
            'sat_loss_factor': 1,
            'B' : 200,
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
            'early_stop': 4000,

            'test_set' : False,
}

graphData, atomRange, allFiles = makeData_fastload_batching(data_path=config['data_path'], norm=config['norm'], 
    structure_type_list=config['structure_types'], batch_size=config['B'],nFiles=config['nFiles'], nNodes=config['nNodes'],
    test=config['test_set'], dummy_center=config['dummy_center'])  

model = nodeVAE(nhid=config['nhid'], numNodesF=config['nNodes']+1+config['satellites'], satellites=config['satellites'],
                numNodes=config['nNodes'], atomRange = atomRange, latent_space=config['latent_space'], cond_dim=config['cond_dim'])

optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0)

print("number of params: ", sum(p.numel() for p in model.parameters()))

N = config['nFiles']
B = config['B']
idx = torch.arange(N)
nTrain = int(0.8*N)
nTrain = nTrain - (nTrain % B)
nValid = N-nTrain
trIdx = idx[:nTrain]
trainFiles = [allFiles[idx] for idx in trIdx]
trIdx = trIdx.view(-1,B)
vlIdx = idx[nTrain:]
validFiles = [allFiles[idx] for idx in vlIdx]

graphData_train = graphData[:int(nTrain/B)]
graphData_validation = graphData[int(nTrain/B):]
print("Using ntrain: %d, nValid: %d"%(nTrain,nValid))

trKl = []
trXYZ = []
trElbo = []

vlKl = []
vlXYZ = []
vlElbo = []
early_stop_counter = 0

obj = ml_manager(model, config, trainFiles, config['train_emb'], epochgrid=config['epochgrid'],
                 folderName=config['savedir'])
obj.set_val(validFiles, config['val_emb'])

start_time = time.time()

if obj.load == True:
    a, b = obj.continue_training()
    model.load_state_dict(torch.load(a))
    optimizer.load_state_dict(torch.load(b))

for i in range(obj.start_epoch, obj.start_epoch + config['num_epochs']):
    start_time = time.time()
    model.train()

    elbo_loss_batch = []
    XYZ_batch = []
    kl_batch = []
    for iter, Batch_Data in enumerate(graphData_train):
        for p in model.parameters(): p.grad = None

        mu, logVar, z, XYZPred, klLoc = model(Batch_Data[2], Batch_Data[6])  # nFeatures

        XYZLoss = reconstruction_loss(structures = Batch_Data[0], XYZPreds = XYZPred, config = config, epoch=i, dummy_center=config['dummy_center'])
        elbo_loss = XYZLoss + config['beta']*klLoc
        
        elbo_loss_batch.append(elbo_loss.item())
        XYZ_batch.append(XYZLoss.item())
        kl_batch.append(klLoc.item())

        if i != obj.start_epoch:
            obj.embedding_manager(i, [iter, len(trainFiles) - 1], [Batch_Data[0]],
                              [mu, logVar, z, XYZPred],
                              ['mu', 'logVar', 'z', 'XYZPred'])

        elbo_loss.backward()
        optimizer.step()

    trElbo.append(np.mean(elbo_loss_batch))
    trXYZ.append(np.mean(XYZ_batch))
    trKl.append(np.mean(kl_batch))

    elbo_loss_batch = []
    XYZ_batch = []
    kl_batch = []
    model.eval()
    with torch.no_grad():
        for iter, Batch_Data in enumerate(graphData_validation):

            mu, logVar, z, XYZPred, klLoc = model(Batch_Data[2], Batch_Data[6])  # nFeatures
            mu_inference, logVar_inference, z_inference, XYZPred_inference, klLoc_inference = model("random", Batch_Data[6], sampling=True)  # nFeatures

            if i % 100 == 5 and iter == 0:
                Atoms_off = make_CIF(structures = Batch_Data[0], XYZPreds = XYZPred_inference, config = config, epoch = i, atomRange = 96, inference=True)
                print ("How many atoms off - inference: {}".format(str(Atoms_off)[1:-1]))
                Atoms_off = make_CIF(structures = Batch_Data[0], XYZPreds = XYZPred, config = config, epoch = i, atomRange = 96, inference=False)
                print ("\nHow many atoms off - NO inference: {}".format(str(Atoms_off)[1:-1]))

            XYZLoss = reconstruction_loss(structures = Batch_Data[0], XYZPreds = XYZPred, config = config, epoch=i, dummy_center=config['dummy_center'])
            XYZLoss_inference = reconstruction_loss(structures = Batch_Data[0], XYZPreds = XYZPred_inference, config = config, epoch=i, dummy_center=config['dummy_center'])

            Val_Loss = XYZLoss_inference #+ config['beta']*klLoc

            if i != obj.start_epoch:
                obj.embedding_manager(i, [iter, len(validFiles) - 1], [Batch_Data[0]], [mu, logVar, z, XYZPred, mu_inference, logVar_inference, z_inference, XYZPred_inference],
                                      ['mu', 'logVar', 'z', 'XYZPred', 'mu_inference', 'logVar_inference', 'z_inference', 'XYZPred_inference'],
                                      mode='VALIDATION')

            elbo_loss_batch.append(Val_Loss)
            XYZ_batch.append(XYZLoss.item())
            kl_batch.append(klLoc.item())

    vlElbo.append(np.mean(elbo_loss_batch))
    vlXYZ.append(np.mean(XYZ_batch))
    vlKl.append(np.mean(kl_batch))

    print("\nEpoch: %d"%(i))
    print("TrELBO: %.5f TrXYZ: %.5f TrKL: %.5f"%(trElbo[-1], trXYZ[-1], trKl[-1]))
    print("inference Loss: %.5f VlXYZ: %.5f VlKL: %.5f "%(vlElbo[-1], vlXYZ[-1], vlKl[-1]))
    #print ("How many atoms off: {}".format(str(Atoms_off)[1:-1]))

    if '{:.5f}'.format(vlElbo[-1]) == '{:.5f}'.format(torch.tensor(vlElbo).min().item()):
        early_stop_counter = 0
        obj.weight_manager(model, optimizer, i)
    else:
        early_stop_counter += 1

    obj.loss_manager(
        [trElbo, trXYZ, trKl,
         vlElbo, vlXYZ, vlKl],
        ['trElbo', 'trXYZ', 'trKl',
         'vlElbo', 'vlXYZ', 'vlKl'], i)

    if early_stop_counter == config['early_stop']:  # todo changed to improve with a specific margin
        print('Model has converged!!!')
        print('\nBest model was found at epoch {}'.format(i-config['early_stop']))
        sys.exit()
    
    print ('Time is: '+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +', Took: {:.4}m'.format((time.time() - start_time)/60))
print('\nConvergence criteria was not met.')
print('Best model was found at epoch {}.'.format(np.argmin(vlElbo)))

