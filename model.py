from typing import Any, Optional, Tuple, no_type_check
import torch.nn.functional as F
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.models import resnet18
from torchmetrics import Accuracy
import math
from resnet import BasicBlock, ResidualNet

# from ResNet import ResidualNet
# This part of the code is implemented based on the official Lightning Module's github repository example https://rb.gy/s78ahn
class CbamMaps(LightningModule):
    def __init__(self):
        self.num_classes=4
        self.save_hyperparameters()
        super().__init__()
        self.model = ResidualNet("CBAM", 18, num_classes=self.num_classes)
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze()
        (
            logits,
            feature_maps,
            channel_attention_maps,
            spatial_attention_maps,
        ) = self.forward(x)
        # logits,channel_feature_maps,spatial_feature_maps,channel_attention_maps, spatial_attention_maps = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        acc = accuracy(logits, y)
        self.log(f"train/loss", loss)
        self.log(f"train/acc", acc, prog_bar=True)
        #self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        (
            logits,
            feature_maps,
            channel_attention_maps,
            spatial_attention_maps,
        ) = self.forward(x)
        # logits,channel_feature_maps,spatial_feature_maps,channel_attention_maps, spatial_attention_maps = self.forward(x)
        y = y.squeeze()
        loss = F.cross_entropy(logits, y.long())
        acc = accuracy(logits, y)
        self.log(f"val/loss", loss)
        self.log(f"val/acc", acc, prog_bar=True)
        #self.log("val_acc", self.test_acc(logits, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        (
            logits,
            feature_maps,
            channel_attention_maps,
            spatial_attention_maps,
        ) = self.forward(x)
        # logits,channel_feature_maps,spatial_feature_maps,channel_attention_maps, spatial_attention_maps = self.forward(x)
        y = y.squeeze()
        loss = F.cross_entropy(logits, y.long())
        acc = accuracy(logits, y)
        self.log(f"val/loss", loss)
        self.log(f"val/acc", acc, prog_bar=True)
        #self.log("val_acc", self.test_acc(logits, y))
        return loss

    def configure_optimizers(self) -> Any:
          optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
          return [optimizer], [StepLR(optimizer, step_size=30, gamma=0.5)]
