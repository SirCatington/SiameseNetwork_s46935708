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

from pytorch_metric_learning import distances, losses, miners, reducers

from dataset import DatasetController
from modules import SiameseNetwork
from utils import AddGaussianNoise, plot_training_loss

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
MODEL_SAVE_PATH = "./siamese_best.pth"
EMBEDDER_MODEL_PATH = "./siamese_best.pth"
FINAL_MODEL_PATH = "./siamese_complete.pth"


VALIDATION_SPLIT = 0.2  # 20% of the training data for validation
TEST_SPLIT = 0.2
UPDATE_INTERVAL = 20
VALIDATION_INTERVAL = UPDATE_INTERVAL
EARLY_STOPPING_PATIENCE = 10

TRUNK_LEARNING_RATE = 5e-5
EMBEDDER_LEARNING_RATE = 5e-4
CLASSIFIER_LEARNING_RATE = 1e-3
L2_LAMBDA =  1e-5
MARGIN = 1

BATCH_SIZE = 128
NUM_EPOCHS = 20

SUPPORT_SET_SIZE = 374

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
    transforms.RandomApply([transforms.RandomAffine(degrees=180)], p=0.5),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0, 0.1))], p=0.5),

    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def embedder_validation_loop(model, data_loader, mining_func, loss_func):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(data_loader):                        
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                embeddings = model(images)
                tuple_indices = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, tuple_indices)             
                    
            running_loss += loss.item()

            # Do the same number of iterations as the train loader
            if batch_idx != 0 and (batch_idx+1) % UPDATE_INTERVAL == 0:
                break

    avg_validation_loss = running_loss / UPDATE_INTERVAL
    return avg_validation_loss

def classifier_validation_loop(model, data_loader, loss_func):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch_idx, (img1, img1, labels) in enumerate(data_loader):                        
            img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)

            with autocast():
                emb1 = model(img1)
                emb2 = model(img2)
                predictions = model.classify(emb1, emb2)
                loss = loss_func(predictions, labels)

            running_loss += loss.item()

            # Do the same number of iterations as the train loader
            if batch_idx != 0 and (batch_idx+1) % UPDATE_INTERVAL == 0:
                break

    avg_validation_loss = running_loss / UPDATE_INTERVAL
    return avg_validation_loss


def train_embedder(model, train_loader, validation_loader, triplet_mining_type):
    optimizer = optim.Adam([
        {'params': model.trunk.parameters(), 'lr': TRUNK_LEARNING_RATE},
        {'params': model.embedder.parameters(), 'lr': EMBEDDER_LEARNING_RATE}
    ], weight_decay=L2_LAMBDA)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=8, threshold=1e-3, verbose=True)

    distance = distances.LpDistance()
    loss_func = losses.TripletMarginLoss(margin=MARGIN, distance=distance, reducer=reducers.ThresholdReducer(low=0))
    mining_func = miners.TripletMarginMiner(margin=MARGIN, distance=distance, type_of_triplets=triplet_mining_type)

    val_mining_func = miners.TripletMarginMiner(margin=MARGIN, distance=distance, type_of_triplets="all")
    val_loss_func = losses.TripletMarginLoss(margin=MARGIN, distance=distance, reducer=reducers.MeanReducer())

    scaler = GradScaler()


    # Store history for plot
    training_loss_history = []
    training_batch_history = []

    validation_loss_history = []
    validation_batch_history = []


    print("Starting training...")

    iteration_count = 0
    best_validation_loss = float("inf")
    patience_counter = 0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                embeddings = model(images)
                tuple_indices = mining_func(embeddings, labels)
                loss = loss_func(embeddings, labels, tuple_indices)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            iteration_count += 1

            training_batch_history.append(iteration_count+1)
            training_loss_history.append(loss.item())

            if (iteration_count+1) % UPDATE_INTERVAL == 0:
                avg_loss = running_loss / UPDATE_INTERVAL
                scheduler.step(avg_loss)
                running_loss = 0.0

                avg_validation_loss = embedder_validation_loop(model, validation_loader, val_mining_func, val_loss_func)

                # Store validation loss for plot 
                validation_batch_history.append(iteration_count+1)
                validation_loss_history.append(avg_validation_loss)

                print(
                    "\nEpoch {} | Iteration {}: Avg. Loss = {:.4f}, Number of mined triplets = {} | Time: {}".format(
                        epoch+1, iteration_count+1, avg_loss, mining_func.num_triplets, datetime.datetime.now().time()
                    )
                )

                print(f"Avg Val Loss: {avg_validation_loss:.4f}")
                      
                if avg_validation_loss < best_validation_loss - 1e-3:
                    print("Validation error improved.")
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)

                    best_validation_loss = avg_validation_loss
                    patience_counter = 0
                    
                else:
                    print(f"Validation error did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                    patience_counter += 1

                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        break

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Patience Exceeded. Stopping Early.")
            break
    
    plot_training_loss(f"embedder_{triplet_mining_type}_training_history.png", training_loss_history, training_batch_history, validation_loss_history, validation_batch_history)

    print(f"Embedder ({triplet_mining_type}) Training finished")
    return model


def train_classifier(model, train_loader, validation_loader):
    model.load_state_dict(torch.load(EMBEDDER_MODEL_PATH, map_location=DEVICE))

    for param in model.trunk.parameters():
        param.requires_grad = False
    for param in model.embedder.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=CLASSIFIER_LEARNING_RATE)
    loss_func = nn.BCELoss()

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=8, threshold=1e-3, verbose=True)

    best_validation_loss = float('inf')
    patience_counter = 0

    # Store history for plot
    training_loss_history = []
    training_batch_history = []

    validation_loss_history = []
    validation_batch_history = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        model.trunk.eval()
        model.embedder.eval()
        
        running_loss = 0.0
        iteration_count = 0
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
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

            training_batch_history.append(iteration_count+1)
            training_loss_history.append(loss.item())

            if (iteration_count+1) % UPDATE_INTERVAL == 0:
                avg_loss = running_loss / UPDATE_INTERVAL
                scheduler.step(avg_loss)
                running_loss = 0.0

                avg_validation_loss = classifier_validation_loop(model, validation_loader, loss_func)

                # Store validation loss for plot 
                validation_batch_history.append(iteration_count+1)
                validation_loss_history.append(avg_validation_loss)

                print(
                    "\nEpoch {} | Iteration {}: Avg. Loss = {:.4f} | Time: {}".format(
                        epoch+1, iteration_count+1, avg_loss, datetime.datetime.now().time()
                    )
                )

                print(f"Avg Val Loss: {avg_validation_loss:.4f}")
                      
                if avg_validation_loss < best_validation_loss - 1e-3:
                    print("Validation error improved.")
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)

                    best_validation_loss = avg_validation_loss
                    patience_counter = 0
                    
                else:
                    print(f"Validation error did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                    patience_counter += 1

                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        break

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Patience Exceeded. Stopping Early.")
            break

    print("Classifier Training finished")

    plot_training_loss("classifier_training_history.png", training_loss_history, training_batch_history, validation_loss_history, validation_batch_history)

    return model
            

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Initalise dataset loaders
    controller = DatasetController(CSV_FILE, IMAGE_DIR, test_transformations, train_transformations, TEST_SPLIT, VALIDATION_SPLIT)
    train_dataset, validation_dataset, test_dataset = controller.get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

    # Initalise model
    model = SiameseNetwork().to(DEVICE)
    model = train_embedder(model, train_loader, validation_loader, "semihard")
    model = train_embedder(model, train_loader, validation_loader, "hard")
    model = train_classifier(model, controller)
    


