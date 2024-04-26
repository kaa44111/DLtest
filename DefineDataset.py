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
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Laden des Bildes
        img_name = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(img_name).convert('RGB')  # Konvertieren zu RGB
        
        # Laden der Masken für dieses Bild
        mask_paths = self.mask_files[index * 6 : (index + 1) * 6]
        masks = [Image.open(os.path.join(self.mask_folder, mask_file)).convert('L') for mask_file in mask_paths]
        
        # Split image and masks into patches
        image_patches = self.extract_patches(image)
        mask_patches = [self.extract_patches(mask) for mask in masks]

        # Optional: Transformationen anwenden
        if self.transform:
            image_patches = [self.transform(img_patch) for img_patch in image_patches]
            mask_patches = [[self.transform(mask_patch) for mask_patch in mask_set] for mask_set in mask_patches]
            
        # return image_patches, mask_patches
        # # Stapeln der Masken und Anpassen der Dimension
        mask_patches = [torch.stack(mask_set) for mask_set in mask_patches]  # Stack each set of mask patches
        masks_tensor = torch.stack(mask_patches, dim=0).squeeze(2)  # Remove unnecessary dimension

        return {
            'image': torch.stack(image_patches),
            'masks': masks_tensor,
        }
    
    def extract_patches(self, img):
        """ Extract patches from an image """
        img_width, img_height = img.size
        patches = []
        for i in range(0, img_height, self.patch_size):
            for j in range(0, img_width, self.patch_size):
                patch = img.crop((j, i, j + self.patch_size, i + self.patch_size))
                patches.append(patch)
        return patches

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

def visualize_image_and_patches(image_patches, mask_patches):
    """ Visualize image and patches """
    num_patches = len(image_patches)
    num_mask_sets = len(mask_patches)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(2, num_patches + 1, 1)
    plt.imshow(image_patches[0])  # Show the first patch as a representative of the whole image
    plt.title('Original Image Patch')
    plt.axis('off')
    
    for i, img_patch in enumerate(image_patches):
        plt.subplot(2, num_patches + 1, i + 2)
        plt.imshow(img_patch)
        plt.title(f'Image Patch {i+1}')
        plt.axis('off')
    
    for i in range(num_mask_sets):
        for j, mask_patch in enumerate(mask_patches[i]):
            plt.subplot(num_mask_sets + 1, num_patches, num_mask_sets * num_patches + j + 1)
            plt.imshow(mask_patch, cmap='gray')
            plt.title(f'Mask {i+1} Patch {j+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
# Annahme: dataset ist eine Instanz von CustomDataset
transform_v2 = v2.Compose([
    v2.ToTensor(),
    #v2.ToDtype(torch.float32, scale=True)
    ])

train_dataset = CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, transform=transform_v2)
image_patches, mask_patches = train_dataset[0]  # Index 0 für das erste Bild im Datensatz
#visualize_image_and_patches(image_patches, mask_patches)




# %%
