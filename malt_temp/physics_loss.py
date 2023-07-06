import torch
import torch.nn as nn

import malt_temp.initial_condition as ic
import malt_temp.kinetic_constants as kc


class PhysicsLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_f = nn.MSELoss()

    def forward(self, prediction, gradient) -> torch.Tensor:
        """
        All the derivatives are from Brandao 2016
        :param prediction:
        :param gradient:
        :return:
        """
        # beta related enzyme
        beta, beta_g = prediction[0,0,0], prediction[0,0,1]
        beta_t, betag_t = gradient[0], gradient[1]

        # temperature dependent coefficients
        k_beta = kc.Kbeta0*torch.exp(-kc.Edbeta/(kc.R * kc.Ta)) # Eq 6

        loss_betag_t = ic.MaltVolume * betag_t - (-kc.H_beta * ic.MaltWeight*(beta_g - beta))  # Eq 2
        loss_beta_t = ic.MaltVolume * beta_t - (-kc.H_beta * ic.MaltWeight*(beta_g - beta) - k_beta * beta)  # Eq 4
        return self.loss_f(loss_beta_t + loss_betag_t, 0)
