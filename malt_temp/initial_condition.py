from collections import namedtuple
from malt_temp import mash_data

InitialCondition = namedtuple("InitialCondition",
                              "alpha_adjust grain_alpha_adjust beta_adjust grain_beta_adjust starch_adjust dextrins_adjust")


InitialAlfa = 0                   # U/kg
InitialGrainAlpha = 1588000        # U/kg
InitialBeta = 0                   # U/kg
InitialGrainBeta = 4840000        # U/kg
InitialStarch = 448.40000         # Starch g/kg
InitialDextrins = 82.40000        # Dextrins g/kg
InitialGlucose = 20.40000         # Glucose g/kg
InitialMaltose = 41.20000         # Maltose g/kg
InitialMaltotriose = 0            # Maltotriose g/kg
InitialLimitDextrins = 0          # Limit Dextrins g/kg

Efic = 59.24
Yeld = 59.24
Factor = Yeld/Efic

InitialAlfa_Adjust = Factor*InitialAlfa*mash_data.MaltWeight/(mash_data.water_volume*1000)                        # U/l
InitialGrainAlpha_Adjust = Factor * InitialGrainAlpha * mash_data.MaltWeight / (mash_data.water_volume * 1000)    # U/l
InitialBeta_Adjust = Factor*InitialBeta*mash_data.MaltWeight/(mash_data.water_volume*1000)                   # U/l
InitialGrainBeta_Adjust = Factor*InitialGrainBeta*mash_data.MaltWeight/(mash_data.water_volume*1000)         # U/l
InitialStarch_Adjust = Factor*InitialStarch*mash_data.MaltWeight/(mash_data.water_volume*1000)               # Starch g/l
InitialDextrins_Adjust = Factor*InitialDextrins*mash_data.MaltWeight/(mash_data.water_volume*1000)           # Dextrins g/l
InitialGlucose_Adjust = Factor*InitialGlucose*mash_data.MaltWeight/(mash_data.water_volume*1000)             # Glucose g/l
InitialMaltose_Adjust = Factor*InitialMaltose*mash_data.MaltWeight/(mash_data.water_volume*1000)             # Maltose g/l
InitialMaltotriose_Adjust = Factor*InitialMaltotriose*mash_data.MaltWeight/(mash_data.water_volume*1000)     # Maltotriose g/l
InitialLimitDextrins_Adjust = Factor*InitialLimitDextrins*mash_data.MaltWeight/(mash_data.water_volume*1000) # Limit Dextrins g/l
