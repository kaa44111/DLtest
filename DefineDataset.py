# %%
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset, DataLoader
import config
from torchvision import transforms
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import random_split

# Define Dataset

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, patch_size=60):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.patch_size = patch_size

        # Liste der Dateinamen in den Ordnern und sortieren nach Bildnummer
        self.image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        self.mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
        
        # Überprüfen, ob die Anzahl der Dateien übereinstimmt
        # if len(self.image_files) != len(self.mask_files) / 6:
        #     raise ValueError("Die Anzahl der Bilder und Masken stimmt nicht überein.")
    
    def __len__(self)-> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Laden des Bildes
        img_name = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(img_name).convert('RGB')  # Konvertieren zu RGB
        
        # Laden der Masken für dieses Bild
        mask_paths = self.mask_files[index * 6 : (index + 1) * 6]
        masks = [Image.open(os.path.join(self.mask_folder, mask_file)).convert('L') for mask_file in mask_paths]
        
        # Split image and masks into patches
        image_patches = self.extract_patches(image)

        # Optional: Transformationen anwenden
        if self.transform:
            image_patches = [self.transform(img_patch) for img_patch in image_patches]
            masks = [self.transform(mask) for mask in masks]

        # # Combine the patches into a single 4D tensor (batch_size, channels, height, width)
        image_tensor = torch.stack(image_patches)

        # Stapeln der Masken und Anpassen der Dimension
        masks_tensor = torch.stack(masks, dim=0)  # Erzeugt einen Tensor der Form [6, 1, 60, 60]
        #Entfernen der überflüssigen Dimension (die '1' bei der Kanaldimension)
        masks_tensor = masks_tensor.squeeze(1)  # Ändert die Form zu [6, 60, 60]

        return image_tensor, masks_tensor


    def extract_patches(self, img: Image.Image) -> list:
        """ Extract patches from an image """
        img_width, img_height = img.size
        patches = []
        for i in range(0, img_height - self.patch_size + 1, self.patch_size):
            for j in range(0, img_width - self.patch_size + 1, self.patch_size):
                patch = img.crop((j, i, j + self.patch_size, i + self.patch_size))
                patches.append(patch)
        return patches



# Methoden zur besseren Übersicht der Dataset 

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
        image_tensor,masks_tensor = dataset[index]
        
        image_tensors.append(image_tensor)
        masks_tensors.append(masks_tensor)

    return image_tensors, masks_tensors

def get_dataloaders():
    transform_v2 = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    custom_dataset = CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, transform=transform_v2,patch_size=60)

    # Definieren Sie die Größen für das Training und die Validierung
    dataset_size = len(custom_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Aufteilen des Datensatzes in Trainings- und Validierungsdaten
    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

    # Erstellen der DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    #Creating Dataloaders:
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders
# %%
def visualize_image_and_patches(image_dict):
    image_patches = image_dict

    num_patches = len(image_patches)
    
    plt.figure(figsize=(15, 5))
    

    for i in range(num_patches):
        plt.subplot(1, num_patches + 1, i + 2)
        plt.imshow(image_patches[i].permute(1, 2, 0))  # Umwandlung in das erwartete Format (Höhe, Breite, Kanäle)
        plt.title(f'Bild Patch {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

transform_v2 = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
])

custom_dataset = CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, transform=transform_v2, patch_size=60)

# Abrufen des ersten Bildes und der ersten Maske
first_image, first_masks = custom_dataset[0]

# Anzeigen der Form der abgerufenen Tensors
print("Erste Bildtensor-Form:", first_image.shape)
print("Erste Maskentensor-Form:", first_masks.shape)