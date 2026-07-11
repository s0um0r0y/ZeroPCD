import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet3D(nn.Module):
    """
    Spatial Transformer Network (T-Net) for 3D Input.
    Predicts a 3x3 transformation matrix to align input points.
    """
    def __init__(self, num_points=1024) -> None:
        super(TNet3D, self).__init__()
        self.num_points = num_points
        
        # shared MLP layers mapped per point as per 1D convolutions
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # fully connected layers to regress the 3X3 matrix
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    