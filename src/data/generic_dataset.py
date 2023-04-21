from torch.utils.data import Dataset
from utils import read_images


class GenericDataset(Dataset):
    def __init__(self, dir_path: str, transform=None) -> None:
        super().__init__()
        self.images = read_images(dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> int:
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        return image
