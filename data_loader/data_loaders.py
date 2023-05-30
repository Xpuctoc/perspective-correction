import albumentations as A
from base import BaseDataLoader
from data_loader.dataset import HomographyDataset


transform = A.Compose([
    A.CLAHE(p=0.5),
    A.ColorJitter(p=0.5),
    A.PixelDropout()
])

class HomographyDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = HomographyDataset(self.data_dir, transforms=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
