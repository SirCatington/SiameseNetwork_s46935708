# showing example usage of your trained model. Print out any results and / or provide visu-
# alisations where applicable
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, recall_score

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def squared_euclidean_distance(x1, x2):
    return torch.sum(torch.pow(x1 - x2, 2), dim=1)

def compute_feature_vectors(model, dataset, include_labels=False):
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            features = model.forward_one(imgs.to(DEVICE))
            all_features.append(features)
            all_labels.append(labels.to(DEVICE))

    if (include_labels):
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    else:
        return torch.cat(all_features, dim=0)


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
            melanoma_similarity = model.prediction(test_features.unsqueeze(1), melanoma_features.unsqueeze(0))
            melanoma_similarity = melanoma_similarity.squeeze(-1) # Shape: (test_size, support_set_size)
            
            normal_similarity = model.prediction(test_features.unsqueeze(1), normal_features.unsqueeze(0))
            normal_similarity = normal_similarity.squeeze(-1) # Shape: (test_size, support_set_size)
            
            avg_mel_sim = melanoma_similarity.mean(dim=1) 
            avg_norm_sim = normal_similarity.mean(dim=1)

            predictions = (avg_mel_sim > avg_norm_sim).long().cpu()

            return predictions, test_labels.cpu()
        
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
            
