from turtle import forward
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
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # spatial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # symetric operation : max pooling across all points
        x = torch.max(x, 2, keepdim = True)[0]
        x = x.view(-1, 1024)
        
        # regression network
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # Initialize transformation as an Identity Matrix to ease optimization
        iden = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
            
        matrix = self.fc3(x).view(-1, 3, 3)
        matrix = matrix + iden
        return matrix
    
class PointNetFeat(nn.Module):
    """
    Core PointNet Feature Extractor.
    Outputs global features and the predicted transformation matrix.
    """
    def __init__(self, num_points=1024, global_feat=True) -> None:
        super(PointNetFeat, self).__init__()
        self.stn = TNet3D(num_points)
        
        # shared MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.global_feat = global_feat
        
    def forward(self, x):
        # x shape: [Batch, 3, Num_Points]
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # apply the 3D spatial transformation matrix
        trans = self.stn(x)
        x = x.transpose(2, 1) # [B, N, 3] for matrix multiplication
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1) # Switch back to [B, 3, N]
        
        # Extract local point features
        x = F.relu(self.bn1(self.conv1(x)))
        point_features = x # Keep track of local features [B, 64, N] 
        
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        global_feature = x.view(-1, 1024)
        
        if self.global_feat:
            return global_feature, trans
        
        else:
            # For segmentation tasks, concatenate global features back to local features
            global_feat_repeat = global_feature.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([point_features, global_feat_repeat], 1), trans