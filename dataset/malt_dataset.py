import pandas as pd
from torch.utils.data import Dataset


class MaltDataset(Dataset):
    def __init__(self, filepath, len_input=3, len_output=3):
        df = pd.read_csv(filepath)
        # input
        self.time, self.temp = df['Time'].values, df['Temperature'].values
        # output
        self.alpha_amilase = df['AlfaAmilase'].values
        self.grain_alpha_amilase = df['AlfaAmilase_Grain'].values   # [U/l]
        self.len_input = len_input
        self.len_output = len_output
        # todo run the code on windows machine and see input, output

    def __len__(self):
        return len(self.time)

    def __getitem__(self, idx):
        return idx, self.temp[idx], self.alpha_amilase[idx], self.grain_alpha_amilase[idx]
