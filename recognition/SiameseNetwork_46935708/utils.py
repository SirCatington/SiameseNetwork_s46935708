import kaggle
import os
import torch
import matplotlib as plt

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

def plot_training_loss(plot_path, training_loss_history, training_batch_history, validation_loss_history, validation_batch_history):
    """
    Plots training and validation loss and saves it to a file.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot training and validation loss
    ax.plot(training_batch_history, training_loss_history, 'o-', color='tab:blue', label='Training Loss')
    ax.plot(validation_batch_history, validation_loss_history, '^-', color='tab:red', label='Validation Loss')

    # Set labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss vs. Iteration Count')
    
    ax.grid(True)
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Save figure
    plt.savefig(plot_path)
    plt.close(fig) 
    print(f"Training history plot saved as {plot_path}")


if __name__ == '__main__':
    download_data()