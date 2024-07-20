import torch as t
from torch.utils.data import DataLoader
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data from the csv file and perform a train-test-split
data = pd.read_csv('data.csv')  # Adjust the filename as necessary
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create an instance of our ResNet model
resnet_model = model.ResNet()

# Set up a suitable loss criterion and optimizer
# Using BCEWithLogitsLoss because it is suitable for multi-label classification
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

# Train the model
res = trainer.fit(epochs=50)

# Plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()
