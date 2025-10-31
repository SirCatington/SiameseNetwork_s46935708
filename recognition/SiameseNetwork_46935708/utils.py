import kaggle
import os
import torch
import torch.nn.functional as F

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

def download_data():
    kaggle.api.authenticate()

    dataset_id = 'nischaydnk/isic-2020-jpg-224x224-resized'
    download_path = './data'
    os.makedirs(download_path, exist_ok=True)

    print(f"Downloading dataset '{dataset_id}' to '{download_path}'...")

    kaggle.api.dataset_download_files(dataset_id, path=download_path, unzip=True)

    print("Download complete!")
    print("Files in directory:", os.listdir(download_path))

if __name__ == '__main__':
    download_data()