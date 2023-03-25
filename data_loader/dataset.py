import numpy as np
import torch
import cv2
import os


class HomographyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_test=False, transforms=None):
        self.data_path = data_path
        self.is_test = is_test
        self.transforms = transforms
        self.images = os.listdir(os.path.join(self.data_path, 'images'))
        # self.images.remove('.DS_Store')

        # self.images = []
        # for image_name in os.listdir(os.path.join(self.data_path, 'images')):
        #     if image_name != '.DS_Store':
        #         image_path = os.path.join(self.data_path, 'images', image_name)
        #         image = cv2.imread(image_path)
        #         if image.shape[:2] == (1600, 1200):
        #             self.images.append(image_name)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_name_clear = image_name[:image_name.rfind('.')]  # without .jpg
        image_path = os.path.join(self.data_path, 'images', image_name)
        image = cv2.imread(image_path)  # BGR

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        image = self.preprocess(image)

        if self.is_test:
            return torch.from_numpy(image).float()

        homography_path = os.path.join(self.data_path, 'homographys',
                                       image_name_clear + '.npy')
        homography = np.load(homography_path)
        homography = homography[:, :2].flatten()

        return torch.from_numpy(image).float(), torch.from_numpy(homography).float()

    def __len__(self):
        return len(self.images)

    def preprocess(self, image):
        # image = cv2.resize(image, (512, 512))
        image = image.transpose(2, 0, 1)
        image = (image - 127.5) / 128

        return image
