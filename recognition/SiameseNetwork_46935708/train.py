# The source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import random
from dataset import DatasetController
from modules import SiameseNetwork
from predict import predict, compute_feature_vectors

import datetime

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "./data/train-image/image"
CSV_FILE = "./data/train-metadata.csv"
BEST_BACKBONE_SAVE_PATH = "backbone_best.pth"
BEST_HEAD_SAVE_PATH = "head_best.pth"

VALIDATION_SPLIT = 0.2  # 20% of the training data for validation
TEST_SPLIT = 0.2        # 20% of the total data for testing
VALIDATION_INTERVAL = 1
EARLY_STOPPING_PATIENCE = 2

#LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 2e-6
HEAD_LEARNING_RATE = 1e-3
LEARNING_DECAY = 0.99
L2_LAMBDA =  2e-3
MARGIN = 1
BATCH_SIZE = 16
NUM_EPOCHS = 20

SUPPORT_SET_SIZE = 32

rotation_degrees = 180
translation_fraction = 0.02
scale_range = (0.8, 1.2)
shear_degrees = 17

test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=rotation_degrees)
    ], p=0.5),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=0, shear=(-shear_degrees, shear_degrees, -shear_degrees, shear_degrees))
    ], p=0.5),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=0, scale=scale_range)
    ], p=0.5),
    
    transforms.RandomApply([
        transforms.RandomAffine(degrees=0, translate=(translation_fraction, translation_fraction))
    ], p=0.5),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# test_transformations = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# train_transformations = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

def squared_euclidean_distance(x1, x2):=
    return torch.sum(torch.pow(x1 - x2, 2), dim=1)

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

    cm = confusion_matrix(test_labels, test_predictions)

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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

    #evaluate_on_test_set("current_model/siamese_epoch_4.pth", controller)
    #evaluate_on_test_set("models/siamese_02/siamese_th07.pth", controller)
    #input("pause:")

    # Initalise models
    model = SiameseNetwork().to(DEVICE)
    criterion_bce = nn.BCELoss()
    criterion_triplet = nn.TripletMarginWithDistanceLoss(
        distance_function=squared_euclidean_distance, 
        margin=MARGIN
    )

    optimizer = optim.Adam([
        {'params': model.feature_map.parameters(), 'lr': BACKBONE_LEARNING_RATE},
        {'params': model.linear.parameters(), 'lr': HEAD_LEARNING_RATE}
    ], weight_decay=L2_LAMBDA)

    scheduler = ExponentialLR(optimizer, gamma=LEARNING_DECAY)

    prev_validation_error = 1.0
    patience_counter = 0


    # Store history for plot
    training_loss_history = []
    validation_error_normal_history = []
    validation_error_melanoma_history = []
    epoch_history = []

    # Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train", leave=False)

        for anchor_img, positive_img, negative_img, targets in train_loop:

            anchor_img, positive_img, negative_img = anchor_img.to(DEVICE), positive_img.to(DEVICE), negative_img.to(DEVICE)

            optimizer.zero_grad()

            anchor_vec, positive_vec, negative_vec = model.forward(anchor_img, positive_img, negative_img)

            loss_triplet = criterion_triplet(anchor_vec, positive_vec, negative_vec)

            loss = loss_triplet

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        training_loss_history.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f} | {datetime.datetime.now().time()}")

        scheduler.step()

        # Validation Loop
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                melanoma_features = compute_feature_vectors(model, melanoma_dataset)
                normal_features = compute_feature_vectors(model, normal_dataset)

                validation_predictions, validation_labels = predict(model, melanoma_features, normal_features, validation_dataset, classifier=False)
                validation_predictions = validation_predictions.to(DEVICE)
                validation_labels = validation_labels.to(DEVICE)


                normal_indices = (validation_labels == 0)
                melanoma_indices = (validation_labels == 1)

                total_normal = normal_indices.sum().item()
                if total_normal > 0:
                    correct_normal = (validation_predictions[normal_indices] == validation_labels[normal_indices]).sum().item()
                    validation_error_normal = 1 - correct_normal / total_normal


                total_melanoma = melanoma_indices.sum().item()
                if total_melanoma > 0:
                    correct_melanoma = (validation_predictions[melanoma_indices] == validation_labels[melanoma_indices]).sum().item()
                    validation_error_melanoma = 1 - correct_melanoma / total_melanoma
                
                avg_validation_error = (validation_error_normal + validation_error_melanoma) / 2

        

            # Store val error for plot after training is finished
            
            validation_error_normal_history.append(validation_error_normal)
            validation_error_melanoma_history.append(validation_error_melanoma)
            epoch_history.append(epoch+1)
        
        
            print(f"Val Normal Error: {validation_error_normal:.4f}, Val Melanoma Error: {validation_error_melanoma:.4f}")
            print(f"Avg Val Error: {avg_validation_error:.4f}")

            if avg_validation_error < prev_validation_error:
                prev_validation_error = avg_validation_error
                patience_counter = 0
                print("Validation error improved.")
                torch.save(model.state_dict(), f"./siamese_best.pth")
            else:
                patience_counter += 1
                print(f"Validation error did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Patience Exceeded. Stopping Early.")
                break

    print("Training finished")
    print(f"Best Val Error: {prev_validation_error:.4f}")

    training_epochs_range  = range(1, len(training_loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(training_epochs_range, training_loss_history, 'o-', color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color_normal = 'tab:red'
    color_melanoma = 'tab:green'
    ax2.set_ylabel('Validation Error', color='black')
    ax2.plot(epoch_history, validation_error_normal_history, 's-', color=color_normal, label='Validation Error (Normal)')
    ax2.plot(epoch_history, validation_error_melanoma_history, '^-', color=color_melanoma, label='Validation Error (Melanoma)')
    ax2.tick_params(axis='y')

    plt.title('Training Loss vs. Validation Error')
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()
    print("Training history plot saved as training_history.png")

    #evaluate_on_test_set(BEST_MODEL_SAVE_PATH, controller)


