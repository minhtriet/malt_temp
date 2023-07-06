import torch.nn as nn

import malt_temp.initial_condition as initial_condition

from torch.autograd import functional as autograd_f

class PINN_Model(nn.Module):
    # def __init__(self, nodes, layers, y0: initial_condition.InitialCondition, w_scale, x_scale=1):
    def __init__(self, nodes, layers, y0: initial_condition.Input):
        """
        Parameters
        ----------
        nodes
        layers
        y0:
            Species concentration at the beginning initial condition, including
            [InitialAlpha_Adjust InitialGrainAlpha_Adjust InitialBeta_Adjust
             InitialGrainBeta_Adjust  InitialStarch_Adjust InitialDextrins_Adjust
        """
        super(PINN_Model, self).__init__()

        self.y0 = y0
        # self.w_scale = w_scale
        # self.x_scale = x_scale

        self.activation = nn.GELU()
        self.seq = nn.Sequential()
        self.seq.add_module('fc_1', nn.Linear(2, nodes))
        self.seq.add_module('relu_1', self.activation)
        for i in range(layers):
            self.seq.add_module('fc_' + str(i + 2), nn.Linear(nodes, nodes))
            self.seq.add_module('relu_' + str(i + 2), self.activation)
        self.seq.add_module('fc_last', nn.Linear(nodes, len(self.y0)))
        # self.seq.add_module('relu_last', nn.Softplus())

    def xavier_init(self):
        for m in self._modules['seq']:

            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def constant_init(self, w0):
        for m in self._modules['seq']:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, w0)
                nn.init.constant_(m.bias, w0)

    def forward(self, x):
        # return self.seq(torch.log(x / self.x_scale)) * (x / self.x_scale) * self.w_scale + self.y0
        y = self.seq(x)
        jacobian = autograd_f.jacobian(self.seq, x)
        return y, jacobian
