from nt import replace
import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

def read_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        
        # Read number of vertices and faces
        n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
        
        # Read all (x, y, z) coordinates
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        
        # Read all faces (we won't use faces for PointNet, just the vertices, but good to parse)
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        
        return np.array(verts), np.array(faces)

class PointSampler:
    """
    Uniformly samples N points from the raw vertices.
    Real point clouds (like LiDAR) have varying densities. 
    PointNet needs a fixed input size (e.g., 512 or 1024 points).
    """
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, verts):
        if len(verts) >= self.output_size:
            idx = np.random.choice(len(verts), self.output_size, replace=False)
        else:
            idx = np.random.choice(len(verts), self.output_size, replace=True)
        return verts[idx]