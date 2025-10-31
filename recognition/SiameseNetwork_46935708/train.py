# The source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

from pytorch_metric_learning import distances, losses, miners, reducers

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

import random
from dataset import DatasetController
from modules import SiameseNetwork
from predict import predict, compute_feature_vectors
from utils import AddGaussianNoise

from torch.cuda.amp import GradScaler, autocast

import datetime

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "./data/train-image/image"
CSV_FILE = "./data/train-metadata.csv"
BEST_MODEL_SAVE_PATH = "./siamese_best.pth"
EMBEDDER_MODEL_PATH = "./siamese_best.pth"
FINAL_MODEL_PATH = "./siamese_complete.pth"


VALIDATION_SPLIT = 0.2  # 20% of the training data for validation
TEST_SPLIT = 0.2
UPDATE_INTERVAL = 20
VALIDATION_INTERVAL = UPDATE_INTERVAL
EARLY_STOPPING_PATIENCE = 10

#LEARNING_RATE = 1e-3
TRUNK_LEARNING_RATE = 5e-5
EMBEDDER_LEARNING_RATE = 5e-4
CLASSIFIER_LEARNING_RATE = 1e-3
LEARNING_DECAY = 0.99
L2_LAMBDA =  1e-5
MARGIN = 1

BATCH_SIZE = 128
NUM_EPOCHS = 20

SUPPORT_SET_SIZE = 374

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
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.05, hue=0.01),
    transforms.ToTensor(),

    transforms.RandomApply([AddGaussianNoise(0., 0.02)], p=0.5),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=180)
    ], p=0.5),

    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),

    # transforms.RandomApply([
    #     transforms.RandomAffine(degrees=0, shear=(-shear_degrees, shear_degrees, -shear_degrees, shear_degrees))
    # ], p=0.5),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))
    ], p=0.5),
    
    transforms.RandomApply([
        transforms.RandomAffine(degrees=0, translate=(0, 0.1))
    ], p=0.5),
    # transforms.RandomApply([
    #     transforms.ElasticTransform(alpha=3.0)
    # ], p=0.5),
    # transforms.RandomApply([
    #     transforms.GaussianBlur(kernel_size=(1, 5))
    # ], p=0.5),

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

def squared_euclidean_distance(x1, x2):
    return torch.sum(torch.pow(x1 - x2, 2), dim=1)

def generate_and_plot_tsne(model, dataset, ax, title):
    """
    Compute embeddings, run t-SNE, and plot the result.
    """

    features, labels = compute_feature_vectors(model, dataset, include_labels=True)
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    tsne = TSNE(random_state=42)
    tsne_results = tsne.fit_transform(features_np)

    normal_indices = np.where(labels_np == 0)
    melanoma_indices = np.where(labels_np == 1)

    ax.scatter(tsne_results[normal_indices, 0], tsne_results[normal_indices, 1], 
               label='Normal', alpha=0.7, c='blue', s=10)
    ax.scatter(tsne_results[melanoma_indices, 0], tsne_results[melanoma_indices, 1], 
               label='Melanoma', alpha=0.7, c='red', s=10)
    
    ax.set_title(title)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()
    ax.grid(True)

def plot_roc_curve(y_true, y_scores):
    """
    Plots the ROC curve.
    """
    display = RocCurveDisplay.from_predictions(
        y_true,
        y_scores,
        name="Siamese k-NN Classifier",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc="lower right")
    plt.grid(True)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold value: {optimal_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  - True Positive Rate (TPR): {tpr[optimal_idx]:.4f}")
    print(f"  - False Positive Rate (FPR): {fpr[optimal_idx]:.4f}")



    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Best Threshold ({optimal_threshold:.2f})')
    plt.legend()
    plt.show()

def evaluate_on_test_set(model_path, controller, classifier=False):
    model = SiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    _, validation_dataset, test_dataset = controller.get_datasets()
    
    train_dataset = controller.train_classification

    # fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    # fig.suptitle('t-SNE Visualization of Embeddings', fontsize=20)

    # generate_and_plot_tsne(model, test_dataset, axes[1], 'Test Set Embeddings')
    
    # plt.savefig("tsne_visualization_comparison.png")
    # plt.close()
    # print("Combined t-SNE plot saved as tsne_visualization_comparison.png")

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

    melanoma_features = compute_feature_vectors(model, melanoma_dataset)
    normal_features = compute_feature_vectors(model, normal_dataset)

    if classifier:
        test_probabilities, test_labels = predict(model, melanoma_features, normal_features, test_dataset, classifier=classifier)
        test_labels = test_labels.numpy()
        test_predictions = (test_probabilities >= 0.4682).astype(int)
        val_probabilities, val_labels = predict(model, melanoma_features, normal_features, validation_dataset, classifier=classifier)
        plot_roc_curve(val_labels, val_probabilities)
        plot_roc_curve(test_labels, test_probabilities)
    else:
        test_predictions, test_labels = predict(model, melanoma_features, normal_features, test_dataset, classifier=classifier)
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
    input("pause:")

def train_classifier(model, controller):
    model.load_state_dict(torch.load(EMBEDDER_MODEL_PATH, map_location=DEVICE))

    for param in model.trunk.parameters():
        param.requires_grad = False
    for param in model.embedder.parameters():
        param.requires_grad = False
    
    train_dataset, validation_dataset, _ = controller.get_datasets_for_classifier()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.classifier.parameters(), lr=CLASSIFIER_LEARNING_RATE)
    loss_func = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        model.trunk.eval()
        model.embedder.eval()
        
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
        for img1, img2, labels in train_loop:
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                emb1 = model(img1)
                emb2 = model(img2)

            predictions = model.classify(emb1, emb2)
            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Avg. Training Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
                emb1 = model(img1)
                emb2 = model(img2)
                predictions = model.classify(emb1, emb2)
                val_loss += loss_func(predictions, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Patience Exceeded. Stopping Early.")
                break

    evaluate_on_test_set(FINAL_MODEL_PATH, controller, True)
            


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Initalise dataset loaders
    controller = DatasetController(CSV_FILE, IMAGE_DIR, test_transformations, train_transformations, TEST_SPLIT, VALIDATION_SPLIT)
    train_dataset, validation_dataset, test_dataset = controller.get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

   # evaluate_on_test_set("siamese_complete.pth", controller, classifier=True)
    #evaluate_on_test_set("current_model/siamese_epoch_4.pth", controller)
    #evaluate_on_test_set("models/siamese_02/siamese_th07.pth", controller)
    
    #visualise_augmentations(train_dataset)

    # Initalise model
    model = SiameseNetwork().to(DEVICE)
    train_classifier(model, controller)
    
    optimizer = optim.Adam([
        {'params': model.trunk.parameters(), 'lr': TRUNK_LEARNING_RATE},
        {'params': model.embedder.parameters(), 'lr': EMBEDDER_LEARNING_RATE},
        {'params': model.classifier.parameters(), 'lr': CLASSIFIER_LEARNING_RATE}
    ], weight_decay=L2_LAMBDA)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=8, threshold=1e-3, verbose=True)

    distance = distances.LpDistance()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=MARGIN, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=MARGIN, distance=distance, type_of_triplets="semihard"
    )

    val_mining_func = miners.TripletMarginMiner(
        margin=MARGIN, distance=distance, type_of_triplets="semihard"
    )

    reducer_do_nothing = reducers.DoNothingReducer()
    loss_func_do_nothing = losses.TripletMarginLoss(margin=MARGIN, distance=distance, reducer=reducer_do_nothing)

    scaler = GradScaler()

    best_validation_loss = float("inf")
    patience_counter = 0


    # Store history for plot
    training_loss_history = []
    validation_error_normal_history = []
    validation_error_melanoma_history = []
    validation_error_avg_history = []
    validation_loss_history = []
    batch_history = []

    # Training Loop
    print("Starting training...")
    iteration_count = 0
    running_loss = 0.0
    running_max_loss = -float("inf")
    hard_mode = False
    for epoch in range(NUM_EPOCHS):
        model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):


            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                embeddings = model(images)
                tuple_indices = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, tuple_indices)
                max_loss = loss

                # # loss_all is integer if no triplets were mined
                # if isinstance(loss_all, torch.Tensor):
                #     loss = loss_all[loss_all != 0].mean()
                #     max_loss = loss_all.max()                                
                # else:     
                #     continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            running_max_loss = max(running_max_loss, max_loss.item())


            iteration_count += 1
            if batch_idx == 0:
                continue

            if (iteration_count+1) % UPDATE_INTERVAL == 0:
                avg_loss = running_loss / UPDATE_INTERVAL
                running_loss = 0.0
                running_max_loss = -float("inf")
                scheduler.step(avg_loss)

                print(
                    "\nEpoch {} | Iteration {}: Max. Loss = {:.4f}, Avg. Loss = {:.4f}, Number of mined triplets = {} | Time: {}".format(
                        epoch+1, iteration_count+1, max_loss, avg_loss, mining_func.num_triplets, datetime.datetime.now().time()
                    )
                )
                training_loss_history.append(avg_loss)
                batch_history.append(iteration_count+1)


                 # Validation Loop

                # model.eval()
                # with torch.no_grad():
                #     running_val_loss = 0.0
                #     running_max_val_loss = -float("inf")
                #     num_validation_batches = 0
                #     for val_images, val_labels in validation_loader:                        
                #         val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)

                #         with autocast():
                #             val_embeddings = model(val_images)
                #             val_tuple_indices = val_mining_func(val_embeddings, val_labels)
                #             val_loss_all = loss_func_do_nothing(val_embeddings, val_labels, val_tuple_indices)['loss']['losses']

                #             # val_loss_all is integer if no triplets were mined
                #             if isinstance(val_loss_all, torch.Tensor):
                #                 val_loss = val_loss_all[val_loss_all != 0].mean()
                #                 max_val_loss = val_loss_all.max()                                
                #             else:     
                #                 val_loss = torch.tensor(0.0, device=DEVICE)
                #                 max_val_loss = torch.tensor(0.0, device=DEVICE)                           
                                


                #         running_val_loss += val_loss.item()
                #         running_max_val_loss = max(running_max_val_loss, max_val_loss.item())
                #         num_validation_batches += 1


                #     avg_validation_loss = running_val_loss / num_validation_batches


                #     melanoma_features = compute_feature_vectors(model, melanoma_dataset)
                #     normal_features = compute_feature_vectors(model, normal_dataset)

                #     validation_predictions, validation_labels = predict(model, melanoma_features, normal_features, validation_dataset, classifier=False)
                #     validation_predictions = validation_predictions.to(DEVICE)
                #     validation_labels = validation_labels.to(DEVICE)


                #     normal_indices = (validation_labels == 0)
                #     melanoma_indices = (validation_labels == 1)

                #     total_normal = normal_indices.sum().item()
                #     if total_normal > 0:
                #         correct_normal = (validation_predictions[normal_indices] == validation_labels[normal_indices]).sum().item()
                #         validation_error_normal = 1 - correct_normal / total_normal


                #     total_melanoma = melanoma_indices.sum().item()
                #     if total_melanoma > 0:
                #         correct_melanoma = (validation_predictions[melanoma_indices] == validation_labels[melanoma_indices]).sum().item()
                #         validation_error_melanoma = 1 - correct_melanoma / total_melanoma
                    
                #     avg_validation_error = (validation_error_normal + validation_error_melanoma) / 2

                model.train()


                # Store val error for plot 
                # validation_error_normal_history.append(validation_error_normal)
                # validation_error_melanoma_history.append(validation_error_melanoma)
                # validation_error_avg_history.append(avg_validation_error)
                # validation_loss_history.append(avg_validation_loss)
       
                # print(f"Val Normal Error: {validation_error_normal:.4f}, Val Melanoma Error: {validation_error_melanoma:.4f}")
                # print(f"Avg Val Loss: {avg_validation_loss:.4f}, Avg Val Error: {avg_validation_error:.4f}")
                # print(f"Max. Val Loss: {running_max_val_loss:.4f}, Avg Val Loss: {avg_validation_loss:.4f}")
                      
                if avg_loss < best_validation_loss - 1e-3:
                    best_validation_loss = avg_loss
                    patience_counter = 0
                    print("Validation error improved.")
                    torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
                else:
                    patience_counter += 1
                    print(f"Validation error did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                    if (patience_counter) == EARLY_STOPPING_PATIENCE and not hard_mode:
                        print("Activating Hard Mode")
                        mining_func = miners.TripletMarginMiner(
                            margin=MARGIN, distance=distance, type_of_triplets="hard"
                        )
                        model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, map_location=DEVICE))

                        # Reset scheduler as hard triplets have different loss values
                        best_validation_loss = float("inf")
                        hard_mode = True
                        patience_counter = 0
                        scheduler = ReduceLROnPlateau(optimizer, "min", patience=8, threshold=1e-4, verbose=True)

                        optimizer.param_groups[0]['lr'] = TRUNK_LEARNING_RATE
                        optimizer.param_groups[1]['lr'] = EMBEDDER_LEARNING_RATE
                        optimizer.param_groups[2]['lr'] = CLASSIFIER_LEARNING_RATE

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    break

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Patience Exceeded. Stopping Early.")
            break

    print("Training finished")
    # print(f"Best Val Error: {prev_validation_error:.4f}")


    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(batch_history, training_loss_history, 'o-', color='tab:blue', label='Training Loss')
    # ax1.plot(batch_history, validation_loss_history, '^-', color='tab:red', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # # ax2 = ax1.twinx()  
    # # color_normal = 'tab:red'
    # # color_melanoma = 'tab:green'
    # # ax2.set_ylabel('Validation Error', color='black')
    # # ax2.plot(batch_history, validation_error_normal_history, 's-', color=color_normal, label='Validation Error (Normal)')
    # # ax2.plot(batch_history, validation_error_melanoma_history, '^-', color=color_melanoma, label='Validation Error (Melanoma)')
    # # ax2.plot(batch_history, validation_error_avg_history, 'o-', color='black', label='Validation Error (Avg)')
    # # ax2.tick_params(axis='y')

    plt.title('Training Loss vs. Validation Error')
    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    # # lines2, labels2 = ax2.get_legend_handles_labels()
    # # ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    ax1.legend(lines, labels, loc='upper right')

    plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()
    print("Training history plot saved as training_history.png")

    evaluate_on_test_set(BEST_MODEL_SAVE_PATH, controller)


