from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
import config
from dataset import malt_dataset
from malt_temp import model, initial_condition

# todo: What boundaries of temperature could be used to generate different BC

path_input = Path("dataset").joinpath("Mash_Data.csv")
dataset = malt_dataset.MaltDataset(path_input, len_input=config.INPUT_LENGTH)
dataloader = DataLoader(dataset)

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



idt = 1e5
n_var = 4
x_scale = idt
# y_scale_old = y_true.max(dim=0).values.to(device=device)
# y_scale = torch.Tensor([y_scale_old[0], y_scale_old[2]]).to(device=device)
# w_scale = torch.ones(n_var).to(device=device) * y_scale
# net = model.PINN_Model(nodes=22, layers=1, y0=initial_cond, w_scale=torch.ones(n_var)*y_scale)

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
