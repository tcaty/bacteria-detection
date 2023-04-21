import lightning

from constants import CROPPED_SUBSTRATES_PATH
from torchvision import transforms
from torch.utils.data import DataLoader
from ..generic_dataset import GenericDataset


class SubstratesDataModule(lightning.LightningDataModule):
    def __init__(
        self, batch_size: int = 128, num_workers: int = 0, shuffle: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        # channels, height, width
        self.dims = (1, 64, 64)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train = GenericDataset(
                dir_path=CROPPED_SUBSTRATES_PATH,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, **self.hparams)
