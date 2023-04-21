import torch
import torchvision
import lightning
import torch.optim as optim
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import lr_scheduler

from .discriminator import Discriminator
from .generator import Generator


class WGAN(lightning.LightningModule):
    def __init__(
        self,
        val_imgs_dir_path: str = "",
        G_latent_dims: int = 100,
        G_lr: float = 0.01,
        D_lr: float = 0.01,
        G_optim_frequency: int = 1,
        D_optim_frequency: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.D = Discriminator()
        self.G = Generator(latent_dims=G_latent_dims)

        self.D.apply(self._weights_init)
        self.G.apply(self._weights_init)

        self.z_dims = (G_latent_dims, 1, 1)
        self.val_imgs_dir_path = val_imgs_dir_path

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def configure_optimizers(self):
        D_lr: float = self.hparams.D_lr  # type: ignore
        G_lr: float = self.hparams.G_lr  # type: ignore

        D_optim = optim.RMSprop(self.D.parameters(), lr=D_lr)
        G_optim = optim.RMSprop(self.G.parameters(), lr=G_lr)

        return [D_optim, G_optim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.G(x)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ):
        x = batch
        D_optim, G_optim = self.optimizers()  # type: ignore

        log_dict = {}

        D_optim_frequency: int = self.hparams.D_optim_frequency  # type: ignore
        G_optim_frequency: int = self.hparams.G_optim_frequency  # type: ignore

        if batch_idx % D_optim_frequency == 0:
            z = torch.randn((len(x), *self.z_dims)).cuda()
            loss = -(torch.mean(self.D(x)) - torch.mean(self.D(self.G(z))))
            self.on_training_step_end(D_optim, loss)
            log_dict["D_loss"] = loss.item()

        if batch_idx % G_optim_frequency == 0:
            z = torch.randn((len(x), *self.z_dims)).cuda()
            loss = -torch.mean(self.D(self.G(z)))
            self.on_training_step_end(G_optim, loss)
            log_dict["G_loss"] = loss.item()

        self.log_dict(log_dict, prog_bar=True)

    def on_training_step_end(self, optim, loss: torch.Tensor) -> None:
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

    def on_train_epoch_end(self) -> None:
        logger: TensorBoardLogger = self.logger  # type: ignore
        shape = 7

        validation_z = torch.randn((shape**2, *self.z_dims)).cuda()
        fake_images = self(validation_z)
        grid = torchvision.utils.make_grid(fake_images, nrow=shape)
        logger.experiment.add_image(
            tag=f"WGAN: epoch {self.current_epoch}", img_tensor=grid
        )

        if self.trainer.is_last_batch and self.current_epoch % 10 == 0:
            torchvision.utils.save_image(
                grid, f"{self.val_imgs_dir_path}/epoch_{self.current_epoch}.png"
            )
