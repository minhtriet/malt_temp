import logging

import pandas as pd
import torch
from torch.utils.data import Dataset


class MaltDataset(Dataset):
    """A dataset that contains the envolvement of the concentration of the malt sacharization
    temperature is a constant till now
    """
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        # input
        self.time, self.temp = df['Time'].values, df['Temperature'].values
        # output
        df = df[['AlfaAmilase', 'AlfaAmilase_Grain']]
        self.features_index = {column_name: df.columns.get_loc(column_name) for column_name in df.columns}
        logging.info(f"Features are {self.features_index}")
        # todo scale
        self.y = torch.from_numpy(df.values)

    def __len__(self):
        return len(self.time)-1

    def __getitem__(self, idx):
        return {'x': torch.FloatTensor([idx, self.temp[idx]]),
                'y': self.y[idx]}
