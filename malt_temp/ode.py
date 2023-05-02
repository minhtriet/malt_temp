import math
from collections import namedtuple

import kinetic_constants as kc
import mash_data

Variables = namedtuple("Variables",
                       """
                       alpha_amylase alpha_amylase_grain beta_amylase  
                       beta_amylase_grain starch dextrins glucose
                       maltose maltotriose limit_dextrins
                       """)


class ODE():
    def __init__(self, species_count):
        self.species_count = species_count
        self.dAlphaAmylase_dt, self.dAlphaAmylase_Grain_dt = 0, 0

    def __call__(self, t, x: Variables):
        """
        Define all the ODEs
        Parameters
        ----------
        t: Time series
        x: All the variables in the ODEs
        Returns
        -------
        """
        # Alpha Amylase Kinetic Model
        Kalpha = kc.Kalpha0 * math.exp(-kc.Edaplha / (kc.R * (t + kc.Ta)))
        # 1
        self.dAlphaAmylase_Grain_dt = -kc.Halpha * (mash_data.MaltWeight / mash_data.malt_volume) * (
                x.alpha_amylase_grain - x.alpha_amylase)
        # 3
        self.dAlphaAmylase_dt = (kc.Halpha * (mash_data.MaltWeight / mash_data.MashVolume) * (
                x.alpha_amylase_grain - x.alpha_amylase)) - \
                                (Kalpha * x.alpha_amylase)

        return self.dAlphaAmylase_dt, self.dAlphaAmylase_Grain_dt
