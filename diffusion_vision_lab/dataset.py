
from abc import ABC, abstractmethod
from torchvision import datasets, transforms
import torch

class BaseDataset(ABC, torch.utils.data.Dataset):
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
    





