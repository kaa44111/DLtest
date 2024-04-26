import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import config
from torchvision import transforms
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import random_split


class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        
        # Liste der Dateinamen in den Ordnern und sortieren nach Bildnummer
        self.image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # Überprüfen, ob die Anzahl der Dateien übereinstimmt
        # if len(self.image_files) != len(self.mask_files) / 6:
        #     raise ValueError("Die Anzahl der Bilder und Masken stimmt nicht überein.")
    
    def __len__(self)-> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Laden des Bildes
        img_name = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(img_name).convert('RGB')  # Konvertieren zu RGB
        
        # Laden der Masken für dieses Bild
        mask_files_for_image = self.mask_files[index * 6 : (index + 1) * 6]
        masks = [Image.open(os.path.join(self.mask_folder, mask_file)).convert('L') for mask_file in mask_files_for_image]
        
        # Optional: Transformationen anwenden
        if self.transform:
            image = self.transform(image)
            masks = [self.transform(mask) for mask in masks]
            
       
        # Stapeln der Masken und Anpassen der Dimension
        masks_tensor = torch.stack(masks, dim=0)  # Erzeugt einen Tensor der Form [6, 1, 60, 60]

        # Entfernen der überflüssigen Dimension (die '1' bei der Kanaldimension)
        masks_tensor = masks_tensor.squeeze(1)  # Ändert die Form zu [6, 60, 60]

        return {
            'image': image,
            'masks': masks_tensor,
        }


def extract_all_tensors(dataset):
    """
    Extrahiert alle Tensoren von den Bildern und den jeweils zugehörigen sechs Masken aus dem Dataset.

    Args:
    - dataset (CustomDataset): Das CustomDataset-Objekt.

    Returns:
    - Tuple[List[torch.Tensor], List[List[torch.Tensor]]]: Ein Tupel bestehend aus einer Liste von Bild-Tensoren
      und einer Liste von Listen von Masken-Tensoren für jedes Bild.
    """
    image_tensors = []
    masks_tensors = []

    for index in range(len(dataset)):
        example = dataset[index]
        image_tensor = example['image']
        masks_tensor = example['masks'] # Liste von sechs Masken-Tensoren
        
        image_tensors.append(image_tensor)
        masks_tensors.append(masks_tensor)

    return image_tensors, masks_tensors

def get_dataloaders():
    transform_v2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)])

    train_dataset = CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, transform=transform_v2)

    # Prozentsatz für das Trainingsset ( 80%)
    train_ratio = 0.8

    # Anzahl der Bilder für das Trainings- und Validierungsset
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Teile den Datensatz in Trainings- und Validierungsset auf
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # DataLoader für Trainings- und Validierungsset
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    #Creating Dataloaders:
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders



