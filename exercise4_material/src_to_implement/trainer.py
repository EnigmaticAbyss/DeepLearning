


import os
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit=None,                    # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit if crit else t.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
        self._optim = optim
        if cuda:
            self._model = model.cuda()
            self._crit = self._crit.cuda()

        self.early_stop_num = early_stopping_patience
        self._best_val_loss = float('inf')
        self.epochs_number_imp = 0
        self.train_data = train_dl
        self.validate_test_data = val_test_dl
        self.cuda_status = cuda
        
    
    
    
    
    
    
    
    
    
    
    
            
    def save_checkpoint(self, epoch):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    

 
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), map_location='cuda' if self.cuda_status else None)
        self._model.load_state_dict(ckp['state_dict']) 
       
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}})
    # training on one step or batch of data (optimaztion step as stated in batch size)
    # for every batch loss and output calculatd and backward being done
    # and updated the weights
    # every batch contrubutes to the gradient of it's own optimization step        
    def train_step(self, x, y):
        self._optim.zero_grad()  # reset the gradients

        output = self._model(x)  # propagate through the network

        loss = self._crit(output, y)  # calculate the loss

        loss.backward()  # compute gradient by backward propagation
        self._optim.step()  # update weights
        return loss.item()
    # test on one batch of data
    # no gradient calculated just checking the results with trained model
    # loss for further evaluation of test phase in comparison to train
    
    
    
    
    
    
    def val_test_step(self, x, y):

        output = self._model(x)

        loss = self._crit(output, y)
        return loss.item(), output
    # train wole batches of data or epoch
    def train_epoch(self):
        self._model.train()  # set training mode
        running_loss = 0.0
        # bar line
        for batch in tqdm(self.train_data):
            x, y = batch
            x = x.cuda() if self.cuda_status else x
            y = y.cuda() if self.cuda_status else y
            # inside train step we transfer variables to gpu
            loss = self.train_step(x, y)
            running_loss += loss
        avg_loss = running_loss / len(self.train_data)
        return avg_loss
    #test whole batches of test or epoch
    # loss is get in test for better understanding of performance and visualization
    
    def val_test(self):
        self._model.eval()  # set eval mode
        running_loss = 0.0
        all_together_preds = []
        all_together_labels = []
        with t.no_grad():
            for batch in tqdm(self.validate_test_data):
                x, y = batch
                x = x.cuda() if self.cuda_status else x
                y = y.cuda() if self.cuda_status else y
                loss, preds = self.val_test_step(x, y)
                preds = t.sigmoid(preds)  # Apply sigmoid to get probabilities

                running_loss += loss
                all_together_preds.append(preds.cpu())
                all_together_labels.append(y.cpu())
        # avg loss by number of elements        
        avg_loss = running_loss / len(self.validate_test_data)
        # tensors become one
        all_together_preds = t.cat(all_together_preds)
        all_together_labels = t.cat(all_together_labels)

        # Apply threshold to get binary predictions
        all_together_preds = all_together_preds > 0.5

        F1_mean = f1_score(all_together_labels.numpy(), all_together_preds.numpy(), average='weighted')

        print(f"Validation Loss is for test: {avg_loss}, F1 Score is this for test: {F1_mean}")
        return avg_loss, F1_mean
# train test some epochs until meeting criteria    
    def fit(self, epochc=-1):
        assert self.early_stop_num > 0 or epochc > 0
        epoch_counter = 0
        train_loss_list = []
        valid_losses = []
       
        scheduler = ReduceLROnPlateau(self._optim, mode='min', factor=0.1, patience=5, verbose=True)
        
        while True:
            epoch_counter += 1
            print(f"Epoch {epoch_counter}")
            train_loss = self.train_epoch()
            validation_loss, validation_f1 = self.val_test()
            train_loss_list.append(train_loss)
            valid_losses.append(validation_loss)
            # Adjust learning rate based on validation loss
            if epoch_counter %2 ==0:
                scheduler.step(validation_loss)
            if validation_loss < self._best_val_loss:
                self._best_val_loss = validation_loss
                self.epochs_number_imp = 0
                self.save_checkpoint(epoch_counter)
                
            else:
                self.epochs_number_imp += 1
            if epochc != 0 and epoch_counter >= epochc and epochc > 0:
                break
                   
            if self.early_stop_num != 0 and self.epochs_number_imp >= self.early_stop_num and self.early_stop_num > 0 :
                print("Early stopping triggered. you have reached the limit of staying the same!")
                break
            
 
        return train_loss_list, valid_losses
