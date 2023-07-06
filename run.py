from pathlib import PurePath

import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import malt_dataset
from malt_temp import model, initial_condition

path_input = PurePath("dataset", "Mash_Data.csv")

dataset = malt_dataset.MaltDataset(path_input)
dataloader = DataLoader(dataset)

if torch.backends.mps.is_available():   # todo torch lightning
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)
net = model.PINN_Model(nodes=22, layers=1, y0=dataset.__getitem__(0)['y'])
criterion = MSELoss()
optimizer = Adam(net.parameters(), lr=0.01)
net.train()
for idx, data in enumerate(dataloader):
    x, y = data['x'], data['y']
    optimizer.zero_grad()
    predict, jacobian = net(x)
    loss_data = criterion(y, predict)  # todo move to pytorch lighting
    loss_physic = 7
    optimizer.step()

# todo jacobian tricks
# Their net production rates are much lower than their
# consumption and production rates and thus can be assumed zero. From a mathematical
# perspective 26, the stiffness of the ODEs can be characterized by the largest absolute
# eigenvalues of the Jacobian matrix, i.e., the Jacobian matrix of the reaction source term to the
# species concentrations. QSSA identifies the species that correspond to the relatively large
# eigenvalues of the chemical Jacobian matrix and then approximate the ODEs with differential-
# algebraic equations to reduce the magnitude of the largest eigenvalue of the Jacobian matrix
# and thus the stiffne
