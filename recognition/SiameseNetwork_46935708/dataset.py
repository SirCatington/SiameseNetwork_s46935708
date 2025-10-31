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

        self.malignant_ids = self.df[self.df['target'] == 1]['isic_id'].tolist()
        self.benign_ids = self.df[self.df['target'] == 0]['isic_id'].tolist()

        print(len(self.malignant_ids))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Determine image class
        if index % 2 == 0:
            label = 0.0
            img_id = random.choice(self.benign_ids)
            
        else:
            label = 1.0
            img_id = random.choice(self.malignant_ids)
            
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def sample_melanoma(self, transform=None):
        img_id = random.choice(self.malignant_ids)
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        if transform:
            img = transform(img)

        return img
    
class SiamesePairDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

        self.malignant_ids = self.df[self.df['target'] == 1]['isic_id'].tolist()
        self.benign_ids = self.df[self.df['target'] == 0]['isic_id'].tolist()
        self.labels = self.df['target'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if random.random() < 0.5:
            # Same class pair
            label = 1.0
            
            if random.random() < 0.5: 
                # Benign
                img1_id = random.choice(self.benign_ids)
                img2_id = random.choice(self.benign_ids)
            else: 
                # Malignant
                img1_id = random.choice(self.malignant_ids)
                img2_id = random.choice(self.malignant_ids)
        else:
            # Different class pair
            label = 0.0
            img1_id = random.choice(self.benign_ids)
            img2_id = random.choice(self.malignant_ids)

        img1_path = os.path.join(self.image_dir, f"{img1_id}.jpg")
        img2_path = os.path.join(self.image_dir, f"{img2_id}.jpg")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

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
        self.validation_dataset = SiameseDataset(self.validation_df, image_dir, self.test_transform)
        self.test_dataset = SiameseClassificationDataset(self.test_df, image_dir, self.test_transform)

        self.train_pair_dataset = SiamesePairDataset(self.train_df, image_dir, self.train_transform)
        self.validation_pair_dataset = SiamesePairDataset(self.validation_df, image_dir, self.test_transform)


    # Returns train, validation and test sets
    def get_datasets(self):       
        return self.train_dataset, self.validation_dataset, self.test_dataset
    
    def get_datasets_for_classifier(self):
        return self.train_pair_dataset, self.validation_pair_dataset, self.test_dataset
    
    def get_support_datasets(self, support_set_size):
        mel_support_df = self.train_df[self.train_df['target'] == 1].sample(support_set_size, random_state=42)
        norm_support_df = self.train_df[self.train_df['target'] == 0].sample(support_set_size, random_state=42)
        
        mel_support_dataset = SiameseClassificationDataset(mel_support_df, self.image_dir, self.test_transform)
        norm_support_dataset = SiameseClassificationDataset(norm_support_df, self.image_dir, self.test_transform)

        return mel_support_dataset, norm_support_dataset
    
    def sample_melanoma(self, transform=None):
        return self.train_dataset(transform)
