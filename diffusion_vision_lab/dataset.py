
from abc import ABC, abstractmethod
from torchvision import datasets, transforms
import torch
from typing import Tuple
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np

class BaseDataset(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path

    @abstractmethod
    def load_transformed_dataset(self):
        """
        Must load the dataset and return the augmented version.
        """
        pass


class CelebA(BaseDataset):

    @staticmethod
    def load_transformed_dataset(img_size: int):
        data_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ])

        train_dataset = datasets.CelebA(root='./data', split='train', download=False, transform=data_transforms)

        return train_dataset
    

class NYUDepthV2TrainingDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root_dir: str,
        rgb_ext: str = '.jpg',
        depth_ext: str = '.png',
        size: Tuple[int, int] = (64, 64),
        ):

        self.root_dir = Path(root_dir)
        self.size = size
        
        self.samples = []
        
        for scene_dir in sorted(self.root_dir.iterdir()):
            
            rgb_files = sorted([f for f in scene_dir.iterdir() if f.suffix == rgb_ext])
            
            for rgb_file in rgb_files:
                base_name = rgb_file.stem 
                depth_file = scene_dir / f"{base_name}{depth_ext}"
                
                if depth_file.exists():
                    self.samples.append((rgb_file, depth_file))
                else:
                    print(f"Warning: No depth map found for {rgb_file}")
        
        print(f"Found {len(self.samples)} valid RGB-Depth pairs across {len(list(self.root_dir.iterdir()))} scenes")


    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        rgb_path, depth_path = self.samples[index]
        
        rgb = Image.open(rgb_path).convert('RGB') 
        depth = Image.open(depth_path)

        rgb = TF.resize(rgb, self.size)
        depth = TF.resize(depth, self.size)

        if random.random() < 0.3:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)

        
        rgb = TF.to_tensor(rgb)
        depth = TF.to_tensor(depth)

        rgb = rgb * 2.0 - 1.0       
        depth = depth * 2.0 - 1.0   

        rgb_depth = torch.cat([rgb, depth], dim=0)  

        return rgb_depth
    


class NYUDepthV2Training(BaseDataset):

    @staticmethod
    def load_transformed_dataset(root_dir: str, size=(64, 64)):
        
        return NYUDepthV2TrainingDataset(root_dir, size)



class NYUDepthV2TestDataset(torch.utils.data.Dataset):
    """
    Expected structure:

    root/
        00000_colors.png
        00000_depth.png
        00001_colors.png
        00001_depth.png
        ...
    """

    def __init__(
        self,
        root_dir: str,
        size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
    ):
        self.root = Path(root_dir)
        self.size = size

        self.samples = []

        color_files = sorted(self.root.glob("*_colors.png"))

        for color_path in color_files:
            stem = color_path.stem.replace("_colors", "")
            depth_path = self.root / f"{stem}_depth.png"

            if depth_path.exists():
                self.samples.append((color_path, depth_path))
            else:
                print(f"Warning: missing depth for {color_path}")

        print(f"Found {len(self.samples)} test pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]


        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)

        rgb = TF.resize(rgb, self.size)
        depth = TF.resize(depth, self.size)
    
        rgb = TF.to_tensor(rgb)         
        rgb = rgb * 2 - 1              
    
        depth = np.array(depth).astype(np.float32)  
        depth /= 1000.0                            
        depth = np.clip(depth, 0, 10.0)              
        depth /= 10.0                          
    
        depth = torch.from_numpy(depth).unsqueeze(0) 
        depth = depth * 2 - 1                        
    
        rgb_depth = torch.cat([rgb, depth], dim=0)    
    
        return rgb_depth
    


class NYUDepthV2Test(BaseDataset):

    @staticmethod
    def load_transformed_dataset(root_dir: str, size=(64, 64)):
        
        return NYUDepthV2TestDataset(root_dir, size)