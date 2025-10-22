# the data loader for loading and preprocessing your data

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import random

class SiameseDataset(Dataset):
    """
    PyTorch Dataset for creating image pairs for a Siamese network.
    """
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)

        self.image_dir = image_dir
        self.transform = transform

        self.melanoma_ids = self.df[self.df['target'] == 1]['isic_id'].tolist()
        self.normal_ids = self.df[self.df['target'] == 0]['isic_id'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        if random.random() < 0.5:
            # Pair consists of same image class
            target = 1.0
            
            if random.random() < 0.5:
                img1_id, img2_id = random.sample(self.normal_ids, 2)
            else:
                img1_id, img2_id = random.sample(self.melanoma_ids, 2)
        else:
            # Pair consists of different image classes
            target = 0.0
            
            img1_id = random.choice(self.normal_ids)
            img2_id = random.choice(self.melanoma_ids)


        img1_path = os.path.join(self.image_dir, f"{img1_id}.jpg")
        img2_path = os.path.join(self.image_dir, f"{img2_id}.jpg")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")


        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(target, dtype=torch.float32)
