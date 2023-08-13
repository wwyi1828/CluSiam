from torch.utils.data import Dataset
import torch
import numpy as np 
from itertools import compress
from PIL import Image
import os
from pathlib import Path
import torchvision.transforms as T


class PathData(Dataset):
    def __init__(self,patch_list, augment) -> None:
        super(PathData).__init__()
        self.patch_list = patch_list
        self.augment = augment
    def __getitem__(self,index):
        patch_path = self.patch_list[index]
        sample = Image.open(patch_path)
        view1, view2 = self.augment(sample)
        return view1, view2

    def __len__(self):
        return len(self.patch_list)
