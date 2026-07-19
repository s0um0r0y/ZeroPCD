from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader
from tqdm import tqdm

from src.data.dataset import ModelNet10Dataset
from src.models.pointnet import PointNetFeat, orthogonal_regularizer_loss

class PointNetClassifier(nn.Module):
    """
    This wraps our Feature Extractor and adds the final classification layers.
    It takes the 1024-dimensional global feature and reduces it to 10 classes.
    """
    def __init__(self, num_classes=10) -> None:
        super(PointNetClassifier, self).__init__()
        
        # core feature extractor (outputs a 1024 vector)
        self.feat_extractor = PointNetFeat(global_feat=True)
        
        # fully connector layers for classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # dropout prevents overfitting on small datasets
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        # Extract features and the transformation matrix
        global_feat, trans_matrix = self.feat_extractor(x)
        
        # pass through classification layers
        x = torch.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Return raw class scores (logits) and the trans_matrix for the loss function
        return x, trans_matrix