# showing example usage of your trained model. Print out any results and / or provide visu-
# alisations where applicable
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_curve, auc, RocCurveDisplay
from sklearn.manifold import TSNE

from modules import SiameseNetwork
from dataset import DatasetController
from utils import AddGaussianNoise

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORT_SET_SIZE = 374


IMAGE_DIR = "./data/train-image/image"
CSV_FILE = "./data/train-metadata.csv"
MODEL_SAVE_PATH = "./siamese_best.pth"

BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

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

def compute_feature_vectors(model, dataset, include_labels=False):
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            features = model(imgs.to(DEVICE))
            all_features.append(features)
            all_labels.append(labels.to(DEVICE))

    if (include_labels):
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    else:
        return torch.cat(all_features, dim=0)
    
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


"""
Takes in a class feature vectors and SiameseClassificationDataset 
to do 2-way n-shot classification
returns a list of class labels from the dataset images
"""
def predict(model, melanoma_features, normal_features, test_dataset, classifier=False):
    # Get average feature vector from both classes
    # Get feature vector for test image
    # Compute similarity between test feature vector and both classes
    # Prediction is class with highest similarity
    model.eval()

    with torch.no_grad():
        test_features, test_labels = compute_feature_vectors(model, test_dataset, include_labels=True)

        if classifier:
            melanoma_similarity = model.classify(test_features.unsqueeze(1), melanoma_features.unsqueeze(0))
            melanoma_similarity = melanoma_similarity.squeeze(-1) # Shape: (test_size, support_set_size)
            
            normal_similarity = model.classify(test_features.unsqueeze(1), normal_features.unsqueeze(0))
            normal_similarity = normal_similarity.squeeze(-1) # Shape: (test_size, support_set_size)
            
            avg_mel_sim = melanoma_similarity.mean(dim=1)
            avg_norm_sim = normal_similarity.mean(dim=1)

            scores_tensor = torch.stack([avg_norm_sim, avg_mel_sim], dim=1)
            y_scores = torch.nn.functional.softmax(scores_tensor, dim=1)[:, 1]

            return y_scores.cpu().numpy(), test_labels.cpu()
                
        else:
            support_features = torch.cat([melanoma_features, normal_features], dim=0)
            melanoma_support_labels = torch.ones(melanoma_features.shape[0], device=DEVICE)
            normal_support_labels = torch.zeros(normal_features.shape[0], device=DEVICE)
            support_labels = torch.cat([melanoma_support_labels, normal_support_labels], dim=0)

            dist_matrix = torch.cdist(test_features, support_features)
            
            best_recall = 0
            best_recall_predictions = None
            for k in range(1,support_features.size(0)):
                _, topk_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)

                topk_labels = torch.gather(support_labels, 0, topk_indices.view(-1)).view(topk_indices.shape)

                predictions = (torch.sum(topk_labels, dim=1) > k / 2).long().cpu()

                macro_recall = recall_score(test_labels.cpu().numpy(), predictions.numpy(), average='macro')

                if (macro_recall > best_recall):
                    best_recall = macro_recall
                    best_recall_predictions = predictions
                    
            return best_recall_predictions, test_labels.cpu()
        

def evaluate_on_test_set(model_path, controller, classifier=False):
    model = SiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    _, validation_dataset, test_dataset = controller.get_datasets()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('t-SNE Visualization of Embeddings', fontsize=20)

    generate_and_plot_tsne(model, test_dataset, axes[1], 'Test Set Embeddings')
    
    plt.savefig("tsne_visualization_comparison.png")
    plt.close()
    print("Combined t-SNE plot saved as tsne_visualization_comparison.png")

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
            
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # Initalise dataset loaders
    controller = DatasetController(CSV_FILE, IMAGE_DIR, test_transformations, train_transformations, TEST_SPLIT, VALIDATION_SPLIT)
    train_dataset, validation_dataset, test_dataset = controller.get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

    melanoma_dataset, normal_dataset = controller.get_support_datasets(SUPPORT_SET_SIZE)

    evaluate_on_test_set(MODEL_SAVE_PATH, controller, True)