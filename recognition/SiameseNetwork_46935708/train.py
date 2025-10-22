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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset import DatasetController
from modules import SiameseNetwork
from predict import predict, compute_feature_vectors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-5
BATCH_SIZE = 16
NUM_EPOCHS = 1

VALIDATION_SPLIT = 0.2  # 20% of the training data for validation
TEST_SPLIT = 0.2        # 20% of the total data for testing

SUPPORT_SET_SIZE = 10

IMAGE_DIR = "./data/train-image/image"
CSV_FILE = "./data/train-metadata.csv"
BEST_MODEL_SAVE_PATH = "siamese_best.pth"


test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_transformations = test_transformations

def evaluate_on_test_set(model_path, controller):
    model = SiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    _, _, test_dataset = controller.get_datasets()

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

    melanoma_features = compute_feature_vectors(model, melanoma_dataset)
    normal_features = compute_feature_vectors(model, normal_dataset)

    test_predictions, test_labels = predict(model, melanoma_features, normal_features, test_dataset)
    test_predictions = test_predictions.numpy()
    test_labels = test_labels.numpy()

    cm = confusion_matrix(test_predictions, test_labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Normal', 'Predicted Melanoma'],
                yticklabels=['Actual Normal', 'Actual Melanoma'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png") 
    plt.close()
    print("Confusion matrix plot saved as confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['Normal', 'Melanoma']))

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
 
    # Initalise dataset loaders
    controller = DatasetController(CSV_FILE, IMAGE_DIR, test_transformations, train_transformations, TEST_SPLIT, VALIDATION_SPLIT)
    train_dataset, validation_dataset, test_dataset = controller.get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

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

    evaluate_on_test_set(BEST_MODEL_SAVE_PATH, controller)
