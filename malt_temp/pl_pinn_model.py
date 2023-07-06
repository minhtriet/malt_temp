import lightning.pytorch as pl
import torch
import torch.functional as F

class PLPinnModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # train loop
        x, y = batch
        prediction = self.model(x)
        loss = F.mse_loss(prediction, x)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        prediction = self.model(x)
        test_loss = F.mse_loss(prediction, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
