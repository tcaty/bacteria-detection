import torch
import lightning as pl
import segmentation_models_pytorch as smp


class BacteriasBinarySegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.create_model(
            "Unet",
            "resnet34",
            in_channels=1,
            classes=1,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        return loss_fn(y, y_hat)

    def forward(self, x: torch.Tensor):
        y_hat = self.model(x)
        y_hat = y_hat.sigmoid()
        return y_hat

    def shared_step(self, batch, stage):
        x, y = batch

        y_hat = self.forward(x)

        pred_mask = (y_hat > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), y.long(), mode="binary"
        )

        log_dict = {
            "loss": self.loss(y, y_hat),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.log_dict(
            log_dict,
            prog_bar=True,
        )
        return log_dict

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
