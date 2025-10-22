import kaggle
import os

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