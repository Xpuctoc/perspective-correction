import numpy as np
import torch
import cv2
import os
import albumentations as A


class HomographyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_test=False, transforms=None):
        self.data_path = data_path
        self.is_test = is_test
        self.transforms = transforms
        self.images = os.listdir(os.path.join(self.data_path, 'images'))

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_name_clear = image_name[:image_name.rfind('.')]  # without .jpg
        image_path = os.path.join(self.data_path, 'images', image_name)
        img = cv2.imread(image_path)  # BGR
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        image = self.preprocess(image)

        if self.is_test:
            return torch.from_numpy(image).float()

        homography_path = os.path.join(self.data_path, 'homographys',
                                       image_name_clear + '.npy')
        homography = np.load(homography_path)
        homography = homography.flatten()

        mean = np.array([9.10049482e-01, -5.85413979e-02, 6.12820893e+01,
                         -3.09079822e-03, 8.47585815e-01, 1.43520874e+01,
                         5.25377075e-06, -2.73889932e-04, 9.95821547e-01])

        std = np.array([2.80111082e-01, 2.16029604e-01, 7.01551215e+01,
                        1.44707826e-01, 3.37215873e-01, 3.60872377e+01,
                        4.02565552e-04, 6.16009993e-04, 2.02862867e-02])

        homography = (homography - mean) / std

        # mean_abs = np.array([9.10049482e-01, 1.33448711e-01, 6.12820893e+01,
        #                      6.29014048e-02, 8.47585815e-01, 1.43520874e+01,
        #                      2.12661938e-04, 5.06282876e-04, 9.95821547e-01])
        #
        # homography = homography / mean_abs

        return torch.from_numpy(image).float(), src_img, torch.from_numpy(homography).float()

    def __len__(self):
        return len(self.images)

    def preprocess(self, image):
        normalize = A.Normalize(mean=(0.471, 0.438, 0.405), std=(0.259, 0.251, 0.251))
        # crop = A.CenterCrop(504, 504)  # for dinov2-vit/14
        image = normalize(image=image)['image']
        # image = crop(image=image)['image']
        image = image.transpose(2, 0, 1)
        # image = (image - 127.5) / 128

        return image
