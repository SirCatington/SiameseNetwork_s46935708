# The source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from modules import SiameseNetwork
from dataset import SiameseDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-5
BATCH_SIZE = 16
NUM_EPOCHS = 1

IMAGE_DIR = "./data/train-image/image"
CSV_FILE = "./data/train-metadata.csv"
BEST_MODEL_SAVE_PATH = "siamese_best.pth"


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
 
    # Initalise dataset loader
    train_dataset = SiameseDataset(CSV_FILE, IMAGE_DIR, transformations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


   # Initalise model
    model = SiameseNetwork().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train", leave=False)

        for img1, img2, labels in train_loop:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
 
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}")

    print("Training finished")

    torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
