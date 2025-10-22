# the source code of the components of your model. Each component must be
# implementated as a class or a function

import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    """
    Siamese Network
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.feature_map = nn.Sequential(*list(resnet.children())[:-1])

        self.linear = nn.Sequential(
            nn.Linear(2048, 1, bias=False),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.feature_map(x)
        # Flatten
        x = x.reshape(x.size(0), -1)
        return x
    
    def prediction(self, feature_vec1, feature_vec2):
        l1_distance = torch.abs(feature_vec1 - feature_vec2)
        prediction_vec = self.linear(l1_distance)
       
        return prediction_vec


    def forward(self, img1, img2):
        feature_vec1 = self.forward_one(img1)
        feature_vec2 = self.forward_one(img2)

        output = self.prediction(feature_vec1, feature_vec2)

        return output




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