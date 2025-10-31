# the source code of the components of your model. Each component must be
# implementated as a class or a function

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese Network
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.trunk = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features

        self.embedder= nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1784),
        nn.ReLU(inplace=True),
        nn.Linear(1784, 1536),
        nn.ReLU(inplace=True),
        nn.Linear(1536, 1024),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.trunk(x)
        # Flatten
        x = x.reshape(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        x = self.embedder(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def classify(self, embedding_1, embedding_2):
        dist = torch.abs(embedding_1 - embedding_2)
        similarity_score = self.classifier(dist)
        return similarity_score.squeeze(1)


# Model test
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SiameseNetwork().to(device)

    batch_size = 8
    img1 = torch.randn(batch_size, 3, 224, 224, device=device)
    img2 = torch.randn(batch_size, 3, 224, 224, device=device)


    output = model(img1, img2)

    print(f"Input 1 Shape: {img1.shape}")
    print(f"Input 2 Shape: {img2.shape}")
    print(f"Model Output Shape: {output.shape}")