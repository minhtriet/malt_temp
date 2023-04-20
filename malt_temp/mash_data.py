MaltWeight = 0.05                                            # kg
water_to_malt_ratio = 4                                         # l/kg
water_volume = MaltWeight * water_to_malt_ratio * 10 ** -3       # m^3
GrainVolumeRatio =  0.656 * 10**-3                           # m^3/kg
malt_volume = MaltWeight * GrainVolumeRatio                   # m^3
GrainAbsorptionRate = 1.043176 * 10**-3                      # m^3/kg
FinalMashVolume = 0.2
MashVolume = water_volume + malt_volume
FinalWortVolume = water_volume - MaltWeight*GrainAbsorptionRate

