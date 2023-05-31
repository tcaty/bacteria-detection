import lightning

from torchvision import transforms
from torch.utils.data import DataLoader
from ..generic_dataset import GenericDataset


class SubstratesDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        dir_path: str,
        dims: tuple[int, int, int],
        batch_size: int = 128,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dir_path", "dims"])

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        # channels, height, width
        self.dims = dims
        self.dir_path = dir_path

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train = GenericDataset(
                dir_path=self.dir_path,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, **self.hparams)
