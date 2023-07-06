# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import matplotlib.pyplot as plt
import numpy as np

from simulai.optimization import Optimizer
from simulai.residuals import SymbolicOperator

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import random


"""REPRODUCIBILITY"""

Fixed_Seed = 42

torch.manual_seed(Fixed_Seed)
np.random.seed(Fixed_Seed)
random.seed(Fixed_Seed)


"""# Kinetics Constants"""

R = 8.31                        # Universal gas constant
Ta = 273.15                     # Celcius to Kelvin

# Starch Gelatinization
Tu = 315.4 - Ta                 # Un-gelatinezed temperature C - threshold temperature
Tg = 336.5 - Ta                 # Gelatinezed temperature C - threshold temperature

# Enzimes Denaturation    
Kalpha0 = 3.86 * 10**34         # Alpha Amylase Pre-exponential coefficient min^-1    
Edaplha = 2.377 * 10**5         # Alpha Amylase Activation energy J/mol
Halpha = 9.72 * 10**-5          # Dissolution coefficient l/g.min^-1

Kbeta0 = 9.46 * 10**67          # Beta Amylase Pre-exponential coefficient min^-1    
Edbeta = 4.439 * 10**5          # Beta Amylase Activation energy J/mol
Hbeta = 7.57 * 10**-5           # Dissolution coefficient l/g.min^-1

# Starch Conversion
Agmlt0 = 6.42 * 10**9           # Maltotriose Pre-exponential coefficient min^-1 
Agdex0 = 3.77 * 10**10          # Dextrin Pre-exponential coefficient min^-1 
Ealpha = 1.03 * 10**5           # Alpha Amylase Activation - Activation energy J/mol

Bgl0 = 1.62 * 10**40            # Glucose Pre-exponential coefficient min^-1 
Bmal0 = 1.05 * 10**42           # Maltose Pre-exponential coefficient min^-1 
Bldex0 = 1.09 * 10**41          # Limit Dextrins Pre-exponential coefficient min^-1 
Ebeta = 2.93 * 10**5            # Beta Amylase Activation - Activation energy J/mol

Km = 2.8                        # Michaelis-Menten Maltose Constant g/l

"""# Mash Data"""

MaltWeight = 0.05                                            # kg  
#Mash Thickness - liquor-to-grist ratio
# 2 to 4 l/kg
WaterToMaltRatio = 4                                         # l/kg                
WaterVolume = MaltWeight *  WaterToMaltRatio  * 10**-3       # m^3 
GrainVolumeRatio =  0.656 * 10**-3                           # m^3/kg                                             
MaltVolume = MaltWeight * GrainVolumeRatio                   # m^3
GrainAbsorptionRate = 1.043176 * 10**-3                      # m^3/kg
FinalMashVolume = 0.2
MashVolume = WaterVolume + MaltVolume
FinalWortVolume = WaterVolume - MaltWeight*GrainAbsorptionRate

Efic = 59.24
Yeld = 59.24
Factor = Yeld/Efic

Weight_Volume_Rel = MaltWeight / MaltVolume

"""# Initial Conditions"""

InitialAlfa = 0.0                  # U/kg
InitialGrainAlfa = 1_588_000       # U/kg
InitialBeta = 0.0                  # U/kg
InitialGrainBeta = 4_840_000       # U/kg
InitialStarch = 448.4              # Starch g/kg
InitialDextrins = 82.4             # Dextrins g/kg
InitialGlucose = 20.4              # Glucose g/kg
InitialMaltose = 41.2              # Maltose g/kg
InitialMaltotriose = 0.0           # Maltotriose g/kg
InitialLimitDextrins = 0.0         # Limit Dextrins g/kg

InitialAlfa = Factor*InitialAlfa*MaltWeight/(WaterVolume*1000)                        # U/l
InitialGrainAlfa = Factor*InitialGrainAlfa*MaltWeight/(WaterVolume*1000)              # U/l
InitialBeta = Factor*InitialBeta*MaltWeight/(WaterVolume*1000)                        # U/l
InitialGrainBeta = Factor*InitialGrainBeta*MaltWeight/(WaterVolume*1000)              # U/l
InitialStarch = Factor*InitialStarch*MaltWeight/(WaterVolume*1000)                    # Starch g/l
InitialDextrins = Factor*InitialDextrins*MaltWeight/(WaterVolume*1000)                # Dextrins g/l
InitialGlucose = Factor*InitialGlucose*MaltWeight/(WaterVolume*1000)                  # Glucose g/l
InitialMaltose = Factor*InitialMaltose*MaltWeight/(WaterVolume*1000)                  # Maltose g/l
InitialMaltotriose = Factor*InitialMaltotriose*MaltWeight/(WaterVolume*1000)          # Maltotriose g/l
InitialLimitDextrins = Factor*InitialLimitDextrins*MaltWeight/(WaterVolume*1000)      # Limit Dextrins g/l

state_t = np.array([InitialAlfa, InitialGrainAlfa, 
                      InitialBeta, InitialGrainBeta, 
                      InitialStarch, 
                      InitialDextrins, InitialGlucose, InitialMaltose, InitialMaltotriose, InitialLimitDextrins])

"""# Calculate Kinetics Values (Temperature Dependent)"""


T = 64.0

"""Enzimes Denaturation """

def Kalpha(t:torch.Tensor) -> torch.Tensor:
  Kalpha = (Kalpha0) * math.exp(-Edaplha / (R * (T + Ta)))
  return Kalpha

def Kbeta(t:torch.Tensor) -> torch.Tensor:
  Kbeta = (Kbeta0) * math.exp(-Edbeta / (R * (T + Ta)))
  return Kbeta

"""Starch Gelatinization


"""

def u(t:torch.Tensor) -> torch.Tensor:
  if T < Tu:
    u = 1
  if T>=Tu and T<=Tg:
    u = (((Tg - T)**2) * ((3 * Tu) - Tg - (2 * T)) / ((Tu - Tg)**3))
  if T>Tg:
    u = 0
  return u

"""Alpha Amylase Enzymatic Reactions"""

# Maltotriose
def Agmlt(t:torch.Tensor) -> torch.Tensor:
  Agmlt = (Agmlt0) * math.exp(-Ealpha / (R * (T + Ta)))
  return Agmlt

# Dextrins
def Agdex(t:torch.Tensor) -> torch.Tensor:
  Agdex = (Agdex0) * math.exp(-Ealpha / (R * (T + Ta)))
  return Agdex

"""Beta Amylase Enzymatic Reactions"""

# Glucose
def Bgl(t:torch.Tensor) -> torch.Tensor:
  Bgl = (Bgl0) * math.exp(-Ebeta / (R * (T + Ta)))
  return Bgl

# Maltose
def Bmal(t:torch.Tensor) -> torch.Tensor:
  Bmal =(Bmal0) * math.exp(-Ebeta / (R * (T + Ta)))
  return Bmal

# Limit Dextrins
def Bldex(t:torch.Tensor) -> torch.Tensor:
  Bldex = (Bldex0) * math.exp(-Ebeta / (R * (T + Ta)))
  return Bldex


# This is a very basic script for exploring the PDE solving via
# feedforward fully-connected neural networks

"""    Variables    """
N = 100
n = 100
Evaluation_Steps = 8_000

def Delta_t(iteration_number):
    if iteration_number<=1:
        Delta_t = 1e-4
    elif iteration_number>1 and iteration_number<=200:
        Delta_t = 1e-3
    else:
        Delta_t = 1e-2
    print("\n Delta t:", Delta_t)
    return Delta_t

n_epochs_ini = 3_000    # Maximum number of iterations for ADAM
n_epochs_min = 200      # Minimum number of iterations for ADAM
Epoch_Tau = 3.0         # Number o Epochs Decay
lr = 1e-3               # Initial learning rate for the ADAM algorithm
def Epoch_Decay(iteration_number):
    if iteration_number<100:
        n_epochs_iter = n_epochs_ini*(np.exp(-iteration_number/Epoch_Tau))
        n_epochs = int(max(n_epochs_iter, n_epochs_min))
    else:
        n_epochs = n_epochs_min
    print("N Epochs:", n_epochs)
    print("Iteration:", iteration_number)
    return n_epochs

def scaling(num):
    orders  = np.floor(np.log10(np.abs(num))).astype(float)
    ten_to_the_power = 5*np.power(10, np.where(num == 0, 0, orders ))
    return ten_to_the_power

def adaptative_weights_residual(num):
    orders = np.floor(np.log10(np.abs(num))).astype(float)
    ten_to_the_negative_power = np.where(num == 0, 1, np.power(10.0, -2*orders))
    return ten_to_the_negative_power

"""# The expressions we aim at minimizing"""

# AlfaAmilase Wort
f_Aw = "D(Aw, t) -((Halpha * Weight_Volume_Rel * (Ag - Aw)) - ((Kalpha(t) * Aw)))"

# AlfaAmilase Grain
f_Ag = "D(Ag, t) -(-Halpha * Weight_Volume_Rel * (Ag - Aw))"

# BetaAmilase Wort
f_Bw = "D(Bw, t) -((Hbeta * Weight_Volume_Rel * (Bg -Bw)) - ((Kbeta(t) * Bw)))"

# BetaAmilase Grain
f_Bg = "D(Bg, t) -(-Hbeta * Weight_Volume_Rel * (Bg - Bw))"

# Starch
f_Sta = "D(Sta, t) -(-Aw * (Sta - (u(t) * Sta)) * (((27 / 28) * Agmlt(t)) + Agdex(t)))"

# Dextrins 
f_De = "D(De, t) - ((Aw * (Sta - (u(t) * Sta)) * Agdex(t)) - (Bw * De * ((9 / 10) * Bgl(t) + (18 / 19) * (Bmal(t) / (Km + De)) + Bldex(t))))"

# Glucose
f_Glu = "D(Glu, t) -(Bgl(t)  * Bw * De)"

# Maltose 
f_Mal = "D(Mal, t) -((Bmal(t) * Bw * De) / (Km + De))"

# Maltotriose
f_Matri = "D(Matri, t) -(Agmlt(t) * Aw * (Sta - (u(t) * Sta)))"

# Limit Dextrins
f_LDe = "D(LDe, t) -(Bldex(t) * Bw * De)"

input_labels = ["t"]
output_labels = ["Aw", "Ag", "Bw", "Bg", "Sta", "De", "Glu", "Mal", "Matri", "LDe"]

n_inputs = len(input_labels)
n_outputs = len(output_labels)

state_t_old = np.array([state_t])
 
depth = 3
width = 50
activations_funct = "tanh"
scale_factors = scaling(state_t)

def model():
  global scale_factors
  from simulai.regression import SLFNN, ConvexDenseNetwork
  from simulai.models import ImprovedDenseNetwork

  #scale_factors = scaling(state_t)
  #np.array([1e+04, 1e+04, 1e+04, 1e+04, 1e+02, 1e+01, 1e+01, 1e+01, 1e+01, 1e+01])
  scale_factors = np.array([5.0e+03, 5.0e+05, 5.0e+03, 5.0e+06, 5.0e+02, 5.0e+01, 5.0e+00, 5.0e+01, 5.0e+00, 5.0e+00])

  # Configuration for the fully-connected network
  config = {
      "layers_units": depth * [width],               # Hidden layers
      "activations": activations_funct,
      "input_size": n_inputs,
      "output_size": n_outputs,
      "name": "net"}
      
  #Instantiating and training the surrogate model
  densenet = ConvexDenseNetwork(**config)
  encoder_u = SLFNN(input_size=1, output_size=width, activation=activations_funct)
  encoder_v = SLFNN(input_size=1, output_size=width, activation=activations_funct)

  class ScaledImprovedDenseNetwork(ImprovedDenseNetwork):
    
    def __init__(self, network=None, encoder_u=None, encoder_v=None, devices="gpu", scale_factors=None):
        
        super(ScaledImprovedDenseNetwork, self).__init__(network=densenet, encoder_u=encoder_u, encoder_v=encoder_v, devices="gpu")
        self.scale_factors = torch.from_numpy(scale_factors.astype("float32")).to(self.device)
        
    
    def forward(self, input_data=None):
        
        return super().forward(input_data)*self.scale_factors
    
  net = ScaledImprovedDenseNetwork(network=densenet, encoder_u=encoder_u, encoder_v=encoder_v, devices="gpu", scale_factors=scale_factors)

  # It prints a summary of the network features
  # net.summary()
      
  return net


net = model()

optimizer_config = {"lr": lr}
optimizer = Optimizer("adam", params=optimizer_config)

residual = SymbolicOperator(
    expressions= [f_Aw, f_Ag, f_Bw, f_Bg, f_Sta, f_De, f_Glu, f_Mal, f_Matri, f_LDe],
    input_vars = input_labels,
    output_vars= output_labels,
    function=net,
    constants={"Weight_Volume_Rel": Weight_Volume_Rel,
               "Halpha": Halpha, 
               "Hbeta": Hbeta,
               "Km": Km},     
    external_functions={"Kalpha": Kalpha,
                        "Kbeta": Kbeta,
                        "u": u,
                        "Agmlt": Agmlt,
                        "Agdex": Agdex,
                        "Bgl": Bgl,
                        "Bmal": Bmal,
                        "Bldex": Bldex},
    engine="torch",
    device="gpu",
)

time_plot = np.empty((0,1), dtype=float)
approximated_data_plot = np.empty((0,n_outputs), dtype=float)
time_eval_plot = np.empty((0,1), dtype=float)

Next_Time = 0 

### A partir daqui rodamos os deltas de tempo
for it in range (0, Evaluation_Steps, 1):
    
    get_Delta_t = Delta_t(it)
    time_train = np.linspace(0, get_Delta_t, n)[:, None]
    time_eval  = np.linspace(0, get_Delta_t, n)[:, None]
    
    initial_state = np.array([state_t])
    new_weights_residual = adaptative_weights_residual(state_t)
    params = {
        "residual": residual,
        "initial_input": np.array([0])[:, None],
        "initial_state": initial_state,
        "weights_residual": [5e-03, 5e-05, 5e-03, 5e-06, 5e+02, 5e+02, 5e+02, 5e+02, 5e+02, 5e+02],
        #[1e-04, 1e-04, 1e-04, 1e-04, 1e+02, 1e+02, 1e+02, 1e+02, 1e+02, 1e+02],
        #[5.0e+03, 5.0e+05, 5.0e+03, 5.0e+06, 5.0e+02, 5.0e+01, 5.0e+00, 5.0e+01, 5.0e+00, 5.0e+00]
        "initial_penalty": 1e6,
    }
    
    get_n_epochs = Epoch_Decay(it)
    optimizer.fit(
        op=net,
        input_data=time_train,
        n_epochs= get_n_epochs,
        loss="pirmse",
        params=params,
        device="gpu",
    )
    
    
    from simulai.optimization import ScipyInterface
    from simulai.optimization import PIRMSELoss
    
    loss_instance = PIRMSELoss(operator=net)
    
    optimizer_lbfgs = ScipyInterface(
        fun=net,
        optimizer="L-BFGS-B",
        loss=loss_instance,
        loss_config=params,
        optimizer_config={
            "options": {
                "maxiter": 50000,
                "maxfun":  50000,
                "maxcor":  50,
                "maxls":   50,
                "ftol": 1.0 * np.finfo(float).eps,
                "eps": 1e-6,}
            },
        )
    
    optimizer_lbfgs.fit(input_data=time_train)
    
    #Evaluation in training dataset
    approximated_data = net.eval(input_data=time_eval)
    state_t = approximated_data[-1]
    
    time_eval_plot  = np.linspace(Next_Time, Next_Time + get_Delta_t, N)[:, None]
    time_plot = np.vstack((time_plot,time_eval_plot))
    approximated_data_plot = np.vstack((approximated_data_plot, approximated_data))
    Next_Time = Next_Time + get_Delta_t
    
    Dataframe_Labels = ['AlfaAmilase', 'AlfaAmilase_Grain', 'BetaAmilase', 'BetaAmilase_grain', 
                        'Starch', 'Dextrins', 'Glucose', 'Maltose',
                        'Maltotriose', 'Limit_Dextrins']
    
    df = pd.DataFrame(approximated_data_plot , columns = Dataframe_Labels)
    
    delta_state_t =  state_t - state_t_old
    grad_state_t = delta_state_t/get_Delta_t
    state_t_old = state_t
    
    scale_factors = scaling(state_t)
    
    if it % 100 == 0:
        ## PINN Charts
        plt.plot(time_plot, df['AlfaAmilase']/1000, label="AlfaAmilase")
        plt.title('Alfa Amilase')
        plt.xlabel('Time (min)')
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()
        
        plt.plot(time_plot, df['AlfaAmilase_Grain']/1000, label="AlfaAmilase_Grain")
        plt.title('Alfa Amilase')
        plt.xlabel('Time (min)')
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()
        
        plt.plot(time_plot, df['BetaAmilase']/1000, label="BetaAmilase")
        plt.title('Beta Amilase')
        plt.xlabel('Time (min)')
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()
        
        plt.plot(time_plot, df['BetaAmilase_grain']/1000, label="BetaAmilase_grain")
        plt.title('Beta Amilase')
        plt.xlabel('Time (min)')
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()
        
        plt.plot(time_plot, df['Starch'], label="Starch")
        plt.plot(time_plot, df['Dextrins'], label="Dextrins")
        plt.plot(time_plot, df['Glucose'], label="Glucose")
        plt.plot(time_plot, df['Maltose'], label="Maltose")
        plt.plot(time_plot, df['Maltotriose'], label="Maltotriose")
        plt.plot(time_plot, df['Limit_Dextrins'], label="Limit_Dextrins")
        plt.title('Sugars')
        plt.xlabel('Time (min)')
        plt.legend()
        plt.show()
    
      
"""     Generating a More Usable Dataset    """
time_plot = time_plot.flatten()
df_time = pd.DataFrame({'Time': time_plot})
df_num_sol = pd.DataFrame(approximated_data_plot , columns = Dataframe_Labels)
Results = pd.concat([df_time,df_num_sol], axis=1)

"""     Export Results for PINN Performance Evaluation    """

Results.to_csv("Mash_PINN.csv", index=False)
