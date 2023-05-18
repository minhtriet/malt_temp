import torch
import config
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

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
        self.input = torch.Tensor(df[input_columns].values)
        # output
        df_output = df.drop(input_columns, axis=1)
        self.output = torch.Tensor(df_output.values)
        self.output_to_index = dict(zip(df_output.columns, range(len(df_output.columns))))

    def __len__(self):
        return self.input.shape[0]-self.len_input

    def __getitem__(self, idx):
        return self.input[idx:idx+self.len_input], \
            self.output[idx:idx+self.len_input, [self.output_to_index['BetaAmilase'],
                                                 self.output_to_index['BetaAmilase_grain']]]
