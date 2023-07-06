from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import config
import logging
import torch

class MaltDataset(Dataset):
    def __init__(self, df, len_input):
        """
        :param filepath: A df that contains the columns
            Time,Temperature,AlfaAmilase,AlfaAmilase_Grain,BetaAmilase,BetaAmilase_grain,
            Starch,Dextrins,Glucose,Maltose,Maltotriose,Limit_Dextrins,SolidosFermentaveis,
            SolidosNaoFermentaveis,SolidosTotais,PercFermentaveis,Extrato,MashingEfficiency,Dp1,Dp2,Dp3,Dp4Plus
        :param len_input: Length of each slice of the dataset to feed to the nn
        """
        # scaling
        self.scaler = StandardScaler()
        n_train = int(config.TRAIN_RATIO * len(df))
        df.iloc[:n_train] = self.scaler.fit_transform(df.iloc[:n_train])
        df.iloc[n_train:] = self.scaler.transform(df.iloc[n_train:])
        self.len_input = len_input
        input_columns = ['Time', 'Temperature']
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
