import torch
import random
import numpy as np
from random import randint

class RandomCrop3D(torch.nn.Module):
    def __init__(self, size:int = 5, fixed: bool = False):
        super(RandomCrop3D, self).__init__()
        self.size = size
        self.fixed = fixed

    def forward(self, source, target):  # we assume inputs are always structured like this
        
        source_x_size, target_x_size, x_start = self._get_random_size(source.shape[0], target.shape[0])
        source_y_size, target_y_size, y_start = self._get_random_size(source.shape[1], target.shape[1])
        source_z_size, target_z_size, z_start = self._get_random_size(source.shape[2], target.shape[2])

        #print("source size: ", source_x_size, source_y_size, source_z_size, "from: ", source.shape)
        #print("target size: ", target_x_size, target_y_size, target_z_size, "from: ", target.shape)
        # Do some transformations. Here, we're just passing though the input
        
        return source[...,x_start:source_x_size, y_start:source_y_size, z_start:source_z_size], target[...,x_start*2:target_x_size, y_start*2:target_y_size, z_start*2:target_z_size]

    def _get_random_size(self, source_size: int, target_size:int):
        
        if self.fixed:
            source = round(source_size/self.size)
        else:
            source = randint(round(source_size/self.size), source_size)
        width = randint(0,source_size - source)
        target = source*2
        if target > target_size:
            target = target_size

        return source+width, target+(width*2), width
    
class RandomFlip(torch.nn.Module):
    def forward(self, source: torch.tensor, target:torch.tensor, chance: float|tuple = 0.33) -> torch.tensor:
        #print(source.shape)
        if isinstance(chance, float):
            x_chance = chance
            y_chance = chance
            z_chance = chance
        elif isinstance(chance, tuple):
            x_chance = chance[0]
            y_chance = chance[1]
            z_chance = chance[2]

        if self.get_random(x_chance):
            source = np.flip(source, -1)
            target = np.flip(target, -1)
        if self.get_random(y_chance):
            source = np.flip(source, -2)
            target = np.flip(target, -2)
        if self.get_random(z_chance):
            source = np.flip(source, -3)
            target = np.flip(target, -3)

        return source, target

    def get_random(self, chance:float) -> bool:
        
        return random.random() < chance