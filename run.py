from pathlib import PurePath

import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from dataset import malt_dataset
from malt_temp import model, initial_condition

path_input = PurePath("dataset", "Mash_Data.csv")
mash_data = pd.read_csv(path_input)

ic = initial_condition.InitialCondition(
    time=mash_data.iloc[0]['Time'],
    temperature=mash_data.iloc[0]['Temperature'],
    alpha_adjust=initial_condition.InitialAlfa_Adjust,
    grain_alpha_adjust=initial_condition.InitialGrainAlpha_Adjust,
    beta_adjust=initial_condition.InitialBeta_Adjust,
    grain_beta_adjust=initial_condition.InitialGrainBeta_Adjust
)

dataset = malt_dataset.MaltDataset(path_input, len_input=config.INPUT_LENGTH)
dataloader = DataLoader(dataset)


ic = initial_condition.InitialCondition(
    time=mash_data.iloc[0]['Time'],
    temperature=mash_data.iloc[0]['Temperature'],
    alpha_adjust=initial_condition.InitialAlfa_Adjust,
    grain_alpha_adjust=initial_condition.InitialGrainAlpha_Adjust,
    beta_adjust=initial_condition.InitialBeta_Adjust,
    grain_beta_adjust=initial_condition.InitialGrainBeta_Adjust
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

initial_cond = initial_condition.InitialCondition(alpha_adjust=initial_condition.InitialAlfa_Adjust,
                                                  grain_alpha_adjust=initial_condition.InitialGrainAlpha_Adjust,
                                                  beta_adjust=initial_condition.InitialBeta_Adjust,
                                                  grain_beta_adjust=initial_condition.InitialGrainBeta_Adjust,
                                                  starch_adjust=initial_condition.InitialStarch_Adjust,
                                                  dextrins_adjust=initial_condition.InitialDextrins_Adjust)

net = model.PINN_Model(len_input=config.INPUT_LENGTH, nodes=22, layers=1, y0=initial_cond)
criterion = MSELoss()
optimizer = Adam(net.parameters(), lr=0.01)
net.train()
for idx, data in enumerate(dataloader):
    time, temp, y = data
    optimizer.zero_grad()
    out = net(time)
    loss_data = criterion(y, out)
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
