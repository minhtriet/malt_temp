from pathlib import PurePath

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import malt_dataset
from malt_temp import model, initial_condition, pl_pinn_model

pl.seed_everything(1234, workers=True)

path_input = PurePath("dataset", "Mash_Data.csv")
mash_data = pd.read_csv(path_input)

dataset = malt_dataset.MaltDataset(mash_data.tail(len(mash_data) - 1), len_input=config.INPUT_LENGTH)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

train_dataloader = DataLoader(train_dataset)  # todo try different kind of sampler
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

initial_cond = initial_condition.InitialCondition(time=mash_data.iloc[0]['Time'],
                                                  temperature=mash_data.iloc[0]['Temperature'],
                                                  beta=mash_data.iloc[0]['BetaAmilase'],
                                                  grain_beta=mash_data.iloc[0]['BetaAmilase_grain'],
                                                  alpha_adjust=initial_condition.InitialAlfa_Adjust,
                                                  grain_alpha_adjust=initial_condition.InitialGrainAlpha_Adjust,
                                                  beta_adjust=initial_condition.InitialBeta_Adjust,
                                                  grain_beta_adjust=initial_condition.InitialGrainBeta_Adjust)

pinn_model = model.PINN_Model(nodes=4, layers=1, y0=initial_cond)

pl_model = pl_pinn_model.PLPinnModule(pinn_model)
trainer = pl.Trainer(max_epochs=5)

# train the model
trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# todo jacobian tricks
# Their net production rates are much lower than their
# consumption and production rates and thus can be assumed zero. From a mathematical
# perspective 26, the stiffness of the ODEs can be characterized by the largest absolute
# eigenvalues of the Jacobian matrix, i.e., the Jacobian matrix of the reaction source term to the
# species concentrations. QSSA identifies the species that correspond to the relatively large
# eigenvalues of the chemical Jacobian matrix and then approximate the ODEs with differential-
# algebraic equations to reduce the magnitude of the largest eigenvalue of the Jacobian matrix
# and thus the stiffne
