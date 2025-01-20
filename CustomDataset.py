import zarr
import numpy as np
import torch
from torch.nn.functional import pad
from glob import glob
from torch.utils.data import Dataset
from random import randint

class CustomTomogramDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        source_tomograms: int = 2,
        target_tomograms: int = 1,
        transform:object= None,
        target_transform:object= None,
        random_get:bool = False
    ):
        self.tomograms = sorted(glob(images_path))
        self.target_tomograms = target_tomograms
        self.source_tomograms = source_tomograms
        self.transform = transform
        self.target_transform = target_transform
        self.random_get = random_get

    def __len__(self) -> int:
        return len(self.tomograms)

    def __getitem__(self, idx: int) -> tuple[np.array, np.array]:
        self.f = zarr.open(self.tomograms[idx])
        #print(type(self.f))

        if self.random_get:
            self.source_tomograms = randint(1,2)
            self.target_tomograms = self.source_tomograms - 1

        source = np.array(self.f.get(self.source_tomograms))
        target = np.array(self.f.get(self.target_tomograms))

        if self.transform:
            source, target = self.transform(source, target)

        target = torch.from_numpy(target[np.newaxis,...].copy())
        source = torch.from_numpy(source[np.newaxis,...].copy())
        
        
        source_shape, target_shape = source.shape, target.shape
        padding = ()
        for x in range(len(source_shape)):
            if source_shape[x] == 1:
                padding = (0,0) + padding
            else:
                padding = (0,source_shape[x]*2 - target_shape[x]) + padding

        if [i for i in padding if i > 0]:
            target = pad(target, padding)
        #print("Source: ", source.shape)
        #print("Target: ", target.shape)
        return source, target
    
    def set_tomograms(self, source_tomograms: int, target_tomograms: int) -> None:
        self.source_tomograms = source_tomograms
        self.target_tomograms = target_tomograms

    def get_tomograms(self) -> tuple[int, int]:
        return self.source_tomograms, self.target_tomograms
    
    def get_tensor_range(self) -> tuple[int|float, int|float]:
        self.min_tensor, self.max_tensor = 0,0
        for i in range(len(self.tomograms)):
            for j in self.__getitem__(i):
                if torch.max(j) > self.max_tensor:
                    self.max_tensor = torch.max(j)
                if torch.min(j) < self.min_tensor:
                    self.min_tensor = torch.min(j)
            
        return self.min_tensor.item(), self.max_tensor.item()