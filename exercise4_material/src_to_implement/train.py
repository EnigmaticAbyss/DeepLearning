import torch as t
from torch.utils.data import DataLoader
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Load the data from the csv file and perform a train-test-split
data = pd.read_csv('data.csv', delimiter=';')
# 2 to 8 test to train
# random state will do same form of split everytime so we don't use test data
# as train data accidentaly!!!
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
# Create an instance of our ResNet model
resnet_model = model.ResNet()
# Set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# these two are for transformations and data aumentation
val_dataset = ChallengeDataset(val_data, mode='validate')
train_dataset = ChallengeDataset(train_data, mode='train')

# loading data and test in batch size of given 32 samples
# the  smaller batch size for train makes  difference since it means faster update of params
# as you know train has bpp and faster change for convergence and more updates
# shuffel train for less overfit and not validation set for finding the patterns as no needed for it to be shuffeled
# furthermore shuffel for train help for mini batch gradient working good as in batch we don't have repeated patterns

val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)


# Set up a suitable loss criterion and optimizer
# Using BCEWithLogitsLoss because it is suitable for multi-label classification
# it is logits loss so works with fcn not acrtivaton of sigmoid!!
criterion = t.nn.BCEWithLogitsLoss()
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.001)

# Create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model=resnet_model,
                  crit=criterion,
                  optim=optimizer,
                  train_dl=train_loader,
                  val_test_dl=val_loader,
                  cuda=t.cuda.is_available(),
                  early_stopping_patience=10)
# trainer.restore_checkpoint(046)
# trainer.save_onnx('checkpoint_{:03d}.onnx'.format(46))
# epoch = int(sys.argv[1])
# # trainer = Trainer(model=resnet_model,
# #                   crit=criterion,
# #                   optim=optimizer,
# #                   train_dl=train_loader,
# #                   val_test_dl=val_loader,
# #                   cuda=t.cuda.is_available(),
# #                   early_stopping_patience=10)
# trainer.restore_checkpoint(epoch)
# trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))

# Train the model
res = trainer.fit(epochc=52)

# Plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
