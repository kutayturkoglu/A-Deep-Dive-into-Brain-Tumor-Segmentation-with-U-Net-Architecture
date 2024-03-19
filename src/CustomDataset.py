import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        