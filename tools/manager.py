import numpy as np
import pandas as pd
import os, sys, h5py, torch

class ml_manager():
    def __init__(self, model, config, ids, numfollow, epochgrid=10, shuffle=False, folderName = 'ml_info'):
        """
        config : dict

        ids : 

        numfollow :

        savedir : 

        epochgrid : int or list
        """
 
        if shuffle == True:
            np.random.shuffle(ids)
        
        self.idsTrain = ids[:numfollow]
        self.epochgrid = epochgrid
        self.folderName = folderName
        self.folderEmb = 'Embeddings'
        self.folderModel = 'model_architeture'
        self.folderGenCIFs = 'gen_XYZ'
        self.start_epoch = 0

        self.load = self._folder_setup()  # Create project folder
        with open(self.folderName+'/Architeture.txt', 'w') as f:
            print(model, file=f)
        f.close()

        df = pd.DataFrame.from_dict(config, orient="index")  # Create dataframe 
        df.to_csv(self.folderName+'/'+"config.csv", header=False)  # Save config


    def set_val(self, ids, numfollow):
        self.idsVal = ids[:numfollow]

        return None


    def embedding_manager(self, epoch, itCount, idList, embs, embslist, mode='Train'):
        # Epoch condition
        if epoch % self.epochgrid != 0:  # Checks for specified epoch
            return None
        elif epoch % self.epochgrid == 0 and itCount[0] == 0:  # Created new file dict for this epoch
            self.h5py_dict = {}  # Empty dict to store objects
            for file in embslist:
                if mode.upper() == 'TRAIN':
                    self.item_list = self.idsTrain
                    ph = h5py.File(self.folderName + '/' + self.folderEmb + '/{}_train_{:08}.hdf5'.format(file,epoch),'w')
                elif mode.upper() == 'VALIDATION':
                    self.item_list = self.idsVal
                    ph = h5py.File(self.folderName + '/' + self.folderEmb + '/{}_validation_{:08}.hdf5'.format(file,epoch),'w')
                else:
                    print('In manager class embedding_manager unknown mode given: {}'.format(mode))
                    print('Should be either Train or Validation!')
                    sys.exit()
                self.h5py_dict.update({'{}'.format(file) : ph})
        else:  # Nothing needs to be done
            pass

        self.ph_train = []
        self.id_ph_train = []
        for item in self.item_list:  # Checks if idsTrain are within given iteration
            try:  # if they are, find index
                self.ph_train.append(idList.index(item))
                self.id_ph_train.append(idList[self.ph_train[-1]])
            except ValueError:  # If not continue
                continue

        if self.ph_train == []:
            return None

        for (key, emb) in zip(self.h5py_dict.keys(),embs):  # If there is data to save, save it
            data_ph = []
            #for i in self.ph_train:
            emb_ph = emb.clone()
            data_ph.append(emb_ph.cpu().detach().numpy())  # Need to be able to reshape

            data_ph = np.array(data_ph)

            for id in self.id_ph_train:
                self.h5py_dict[key].create_dataset('{}'.format(id), data=data_ph)

        if itCount[0] == itCount[1]:  # Close open files at the end of the Epoch
            for key in self.h5py_dict.keys():
                self.h5py_dict[key].close()

        return None


    def loss_manager(self, loss, header,epoch):
        if self.start_epoch == 0:
            df = pd.DataFrame(np.array(loss).T)  # Create dataframe
            df.to_csv(self.folderName+'/loss.csv', header=header)  # Save config
        else:
            df = pd.read_csv(self.folderName+'/loss.csv',index_col=0)
            loss = np.array(loss).T

            df2 = pd.DataFrame(loss,columns=header,index=range(self.start_epoch, epoch+1))
            df = pd.concat([df,df2])
            df = df[~df.index.duplicated(keep='last')]
            df.to_csv(self.folderName+'/loss.csv', header=header)
           
        return None



    def weight_manager(self,model,optimizer,epoch):
        #if epoch % self.epochgrid != 0:
        #    return None
        #if not save_model:
        #    return None

        torch.save(model.state_dict(), self.folderName+'/'+self.folderModel+'/'+'model_{:08}.pt'.format(epoch))
        torch.save(optimizer.state_dict(), self.folderName+'/'+self.folderModel+'/'+'optimizer_{:08}.pt'.format(epoch))
        best_model = sorted(os.listdir(self.folderName+'/'+self.folderModel))
        if len(best_model) > 4:
            num = int(best_model[-3][-11:-3])
            os.remove(self.folderName+'/'+self.folderModel+'/'+'model_{:08}.pt'.format(num))
            os.remove(self.folderName+'/'+self.folderModel+'/'+'optimizer_{:08}.pt'.format(num))
        return None


    def continue_training(self):
        print(self.folderName+'/'+self.folderModel)
        best_model = sorted(os.listdir(self.folderName+'/'+self.folderModel))
        print(best_model)
        num = int(best_model[-1][-11:-3])
        self.start_epoch = num
        return self.folderName+'/'+self.folderModel+'/'+'model_{:08}.pt'.format(num), self.folderName+'/'+self.folderModel+'/'+'optimizer_{:08}.pt'.format(num)




    def _folder_setup(self):
        """

        """

        
        if os.path.isdir(self.folderName):
            print('Continue or new')
            answer = self._user_int()
            if answer == 0:
                return True
            else:
                count = 0
                while True:
                    if os.path.isdir(self.folderName+'_{}'.format(count)):
                        count+=1
                    else:
                        self.folderName = self.folderName+'_{}'.format(count)
                        break


            os.mkdir(self.folderName)
            print("{} has been created".format(self.folderName))

            os.mkdir(self.folderName+'/'+self.folderEmb)
            print("{} has been created".format(self.folderName+'/'+self.folderEmb))

            os.mkdir(self.folderName+'/'+self.folderModel)
            print("{} has been created".format(self.folderName+'/'+self.folderModel))

            os.mkdir(self.folderName+'/'+self.folderGenCIFs)
            print("{} has been created".format(self.folderName+'/'+self.folderGenCIFs))

        else:
            os.mkdir(self.folderName)
            print("{} has been created".format(self.folderName))

            os.mkdir(self.folderName+'/'+self.folderEmb)
            print("{} has been created".format(self.folderName+'/'+self.folderEmb))

            os.mkdir(self.folderName+'/'+self.folderModel)
            print("{} has been created".format(self.folderName+'/'+self.folderModel))

            os.mkdir(self.folderName+'/'+self.folderGenCIFs)
            print("{} has been created".format(self.folderName+'/'+self.folderGenCIFs))

        return False


    def _user_int(self):
        print('\nProject folder already exists')
        print('\t0 to proceed from last saved epoch')
        print('\t1 to create new folder')
        while True:
            try:
                answer = int(input())
            except ValueError:
                answer = None
                print('Input can only be 1 or 0')
            if answer == 0:
                break
            elif answer == 1:
                break
            else:
                print('Answer not recognized, should be 0 or 1')


        return answer

