import lightning.pytorch as pl
import torch
import torch.nn as nn

import malt_temp.physics_loss as physics_loss

class PLPinnModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_f = nn.MSELoss()
        self.physics_loss = physics_loss.PhysicsLoss()

    def training_step(self, batch, batch_idx):
        # train loop
        x, y = batch
        prediction, grads = self.model(x)
        prediction_loss = self.loss_f(prediction[0], y)
        physics_loss = self.physics_loss(prediction, grads)
        loss = prediction_loss + physics_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # test loop
        x, y = batch
        prediction, grads = self.model(x)
        val_loss = self.loss_f(prediction[0], y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
