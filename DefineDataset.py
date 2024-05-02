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
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Laden des Bildes
        img_name = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(img_name).convert('RGB')  # Konvertieren zu RGB
        
        # Laden der Masken für dieses Bild
        mask_paths = self.mask_files[index * 6 : (index + 1) * 6]
        masks = [Image.open(os.path.join(self.mask_folder, mask_file)).convert('L') for mask_file in mask_paths]
        
        # Split image and masks into patches
        image_patches = self.extract_patches(image)
        #mask_patches = [self.extract_patches(mask) for mask in masks]

        # Optional: Transformationen anwenden
        if self.transform:
            image_patches = [self.transform(img_patch) for img_patch in image_patches]
            #mask_patches = [[self.transform(mask_patch) for mask_patch in mask_set] for mask_set in mask_patches]
            masks = [self.transform(mask) for mask in masks]

         # Stapeln der Masken und Anpassen der Dimension
        masks_tensor = torch.stack(masks, dim=0)  # Erzeugt einen Tensor der Form [6, 1, 60, 60]

        # Entfernen der überflüssigen Dimension (die '1' bei der Kanaldimension)
        masks_tensor = masks_tensor.squeeze(1)  # Ändert die Form zu [6, 60, 60]

        return {
            'image' : image_patches, 
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
        example = dataset[index]
        image_tensor = example['image']
        masks_tensor = example['masks'] # Liste von sechs Masken-Tensoren
        
        image_tensors.append(image_tensor)
        masks_tensors.append(masks_tensor)

    return image_tensors, masks_tensors

def get_dataloaders(patch_size=60):
    transform_v2 = v2.Compose([
        v2.ToTensor(),
    #v2.ToDtype(torch.float32, scale=True)
    ])

    # Definieren Sie Ihren CustomDataset
    custom_dataset = CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, patch_size=patch_size, transform=transform_v2)

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
   # original_image = image_dict['original']
    image_patches = image_dict#['patches']

    num_patches = len(image_patches)
    
    plt.figure(figsize=(15, 5))
    
    # plt.subplot(1, num_patches + 1, 1)
    # plt.imshow(original_image)  # Originalbild anzeigen
    # plt.title('Original Bild')
    # plt.axis('off')

    for i in range(num_patches):
        plt.subplot(1, num_patches + 1, i + 2)
        plt.imshow(image_patches[i].permute(1, 2, 0))  # Umwandlung in das erwartete Format (Höhe, Breite, Kanäle)
        plt.title(f'Bild Patch {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# %%
# Annahme: dataset ist eine Instanz von CustomDataset
transform_v2 = v2.Compose([
    v2.ToTensor(),
    #v2.ToDtype(torch.float32, scale=True)
    ])

# Beispielaufruf
train_dataset = CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, patch_size=64, transform=transform_v2)
image_patches = train_dataset[0]['image']  # Index 0 für das erste Bild im Datensatz
#visualize_image_and_patches(image_patches)
# example = train_dataset[0]
# # Überprüfe die Form der Maskendaten (Tensor)
# print("Form des Masken-Tensors:", example['masks'].shape)

#train_dataloader = get_dataloaders(patch_size=60)['train']


train_dataloader = DataLoader(train_dataset, batch_size=9, shuffle=False)
batch = next(iter(train_dataloader))

image_patches1 = batch['image']  # image_patches hat die Form (batch_size, num_patches, channels, height, width)
first_image_patches = image_patches1[0]  # Extrahiere die ersten Patches des ersten Bildes
#visualize_image_and_patches(first_image_patches)
# Anzahl der Patches im DataLoader
num_patches_dataloader = len(batch['image'][0])  # Anzahl der Patches im ersten Bild im DataLoader

# Anzahl der Patches im Dataset
num_patches_dataset = len(train_dataset[0]['image'])  # Anzahl der Patches im ersten Bild des Datasets

print("Anzahl der extrahierten Patches im DataLoader:", num_patches_dataloader)
print("Anzahl der extrahierten Patches im Dataset:", num_patches_dataset)
# %%
