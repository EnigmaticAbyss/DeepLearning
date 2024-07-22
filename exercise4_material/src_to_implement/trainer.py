
                    
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self._best_val_loss = float('inf')
        self._epochs_no_improve = 0

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), map_location='cuda' if self._cuda else None)
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
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        self._optim.zero_grad()  # reset the gradients
        x = x.cuda() if self._cuda else x
        y = y.cuda() if self._cuda else y
        output = self._model(x)  # propagate through the network
   
        loss = self._crit(output, y)  # calculate the loss
    
        loss.backward()  # compute gradient by backward propagation
        self._optim.step()  # update weights
        return loss.item()
    
    def val_test_step(self, x, y):
        x = x.cuda() if self._cuda else x
        y = y.cuda() if self._cuda else y
        output = self._model(x)

        loss = self._crit(output, y)
        return loss.item(), output
    
    def train_epoch(self):
        self._model.train()  # set training mode
        running_loss = 0.0
        for batch in tqdm(self._train_dl):
            x, y = batch
            loss = self.train_step(x, y)
            running_loss += loss
        avg_loss = running_loss / len(self._train_dl)
        return avg_loss
    
    def val_test(self):
        self._model.eval()  # set eval mode
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with t.no_grad():
            for batch in tqdm(self._val_test_dl):
                x, y = batch
                loss, preds = self.val_test_step(x, y)
                running_loss += loss
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
        avg_loss = running_loss / len(self._val_test_dl)
        all_preds = t.cat(all_preds)
        all_labels = t.cat(all_labels)
        f1 = f1_score(all_labels, all_preds.argmax(dim=1), average='weighted')
        print(f"Validation Loss: {avg_loss}, F1 Score: {f1}")
        return avg_loss, f1
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses = []
        epoch_counter = 0
        
        while True:
            epoch_counter += 1
            print(f"Epoch {epoch_counter}")
            train_loss = self.train_epoch()
            val_loss, val_f1 = self.val_test()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._epochs_no_improve = 0
                self.save_checkpoint(epoch_counter)
            else:
                self._epochs_no_improve += 1
            
            if self._early_stopping_patience > 0 and self._epochs_no_improve >= self._early_stopping_patience:
                print("Early stopping triggered.")
                break
            
            if epochs > 0 and epoch_counter >= epochs:
                break
        
        return train_losses, val_losses
        
        
        
