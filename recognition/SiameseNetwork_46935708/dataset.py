# the data loader for loading and preprocessing your data

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split


class SiameseDataset(Dataset):
    """
    PyTorch Dataset for creating image pairs for a Siamese network.
    """
    def __init__(self, df, image_dir, transform=None):
        self.df = df

        self.image_dir = image_dir
        self.transform = transform

        self.melanoma_ids = self.df[self.df['target'] == 1]['isic_id'].tolist()
        self.normal_ids = self.df[self.df['target'] == 0]['isic_id'].tolist()

        #self.normal_proportion = len(self.normal_ids) / len(df)
        self.normal_proportion = 0.5

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Determine if the anchor is normal or melanoma
        if random.random() < self.normal_proportion:
            # Anchor is normal
            anchor_id, positive_id = random.sample(self.normal_ids, 2)
            negative_id = random.choice(self.melanoma_ids)
        else:
            # Anchor is melanoma
            anchor_id, positive_id = random.sample(self.melanoma_ids, 2)
            negative_id = random.choice(self.normal_ids)
        
        anchor_path = os.path.join(self.image_dir, f"{anchor_id}.jpg")
        positive_path = os.path.join(self.image_dir, f"{positive_id}.jpg")
        negative_path = os.path.join(self.image_dir, f"{negative_id}.jpg")

        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        target = random.randint(0, 1)

        return anchor_img, positive_img, negative_img, target

class SiameseClassificationDataset(Dataset):
    """
    PyTorch Dataset for classification. Gives an image and its label
    """
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df.iloc[index]['isic_id']
        label = self.df.iloc[index]['target']

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)
    
class DatasetController:
    def __init__(self, csv_path, image_dir, test_transform, train_transform, test_split, validation_split):
        self.df = pd.read_csv(csv_path)

        #self.df, _ = train_test_split(self.df, test_size=0.99, stratify=self.df['target'], random_state=42)
        
        #print(f"Dataset size: {len(self.df)}")

        self.image_dir = image_dir
        self.test_transform = test_transform
        self.train_transform = train_transform

        train_val_df, self.test_df = train_test_split(self.df, test_size=test_split, stratify=self.df['target'], random_state=42)

        self.train_df, self.validation_df = train_test_split(train_val_df, test_size=validation_split, stratify=train_val_df['target'], random_state=42)

        self.train_dataset = SiameseDataset(self.train_df, image_dir, self.train_transform)
        self.validation_dataset = SiameseClassificationDataset(self.validation_df, image_dir, self.test_transform)
        self.test_dataset = SiameseClassificationDataset(self.test_df, image_dir, self.test_transform)


    # Returns train, validation and test sets
    def get_datasets(self):       
        return self.train_dataset, self.validation_dataset, self.test_dataset
    
    def get_support_datasets(self, support_set_size):
        mel_support_df = self.train_df[self.train_df['target'] == 1].sample(support_set_size, random_state=42)
        norm_support_df = self.train_df[self.train_df['target'] == 0].sample(support_set_size, random_state=42)
        
        mel_support_dataset = SiameseClassificationDataset(mel_support_df, self.image_dir, self.test_transform)
        norm_support_dataset = SiameseClassificationDataset(norm_support_df, self.image_dir, self.test_transform)

        return mel_support_dataset, norm_support_dataset
