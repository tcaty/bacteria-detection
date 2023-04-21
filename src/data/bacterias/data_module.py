import lightning
import data.bacterias.transforms as bacterias_transforms

from constants import CROPPED_STANDALONE_BACTERIAS_PATH
from torchvision import transforms
from torch.utils.data import DataLoader
from ..generic_dataset import GenericDataset


class BacteriasDataModule(lightning.LightningDataModule):
    def __init__(
        self, batch_size: int = 128, num_workers: int = 0, shuffle: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                bacterias_transforms.Square(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        # channels, height, width
        self.dims = (1, 64, 64)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train = GenericDataset(
                dir_path=CROPPED_STANDALONE_BACTERIAS_PATH,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, **self.hparams)
