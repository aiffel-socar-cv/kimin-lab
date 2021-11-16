import os
from glob import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


import albumentations as A
from albumentations.pytorch import transforms


class Dataset(Dataset):
    def __init__(self, imgs_dir, mask_dir, transform=None):
        self.img_dir = imgs_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(glob(os.path.join(imgs_dir, '*.jpg')))

        files = []
        for ext in ('*.gif', '*.png', '*.jpg'):
            files.extend(glob(os.path.join(mask_dir, ext)))
        self.masks = sorted(files)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        np_image = np.array(image)
        np_mask = np.array(mask)
        
        if self.transform:
            transformed = self.transform(image=np_image, mask=np_mask)
            np_image = transformed["image"]
            np_mask = transformed["mask"]
            np_mask = np_mask.long()
        
        ret = {
            'img': np_image,
            'label': np_mask,
        }
        
        return ret

if __name__ == '__main__':

    transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=0.5, std=0.5),
                transforms.ToTensorV2()
            ])

    img_dir = os.path.join(os.getcwd(), "accida_segmentation", 'imgs', 'train')
    mask_dir = os.path.join(os.getcwd(), "accida_segmentation", 'labels', 'train')

    train_dataset = Dataset(img_dir, mask_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)

    #For shape test
    for ret in iter(dataloader):
        print(ret['img'].shape, ret['label'].shape, ret['label'].type)