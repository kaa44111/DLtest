import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import config
from torchvision import transforms
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        
        # Liste der Dateinamen in den Ordnern und sortieren nach Bildnummer
        self.image_files = sorted(os.listdir(image_folder), key=self.extract_image_number)
        self.mask_files = sorted(os.listdir(mask_folder), key=self.extract_image_number)
        
        # Überprüfen, ob die Anzahl der Dateien übereinstimmt
        # if len(self.image_files) != len(self.mask_files) / 6:
        #     raise ValueError("Die Anzahl der Bilder und Masken stimmt nicht überein.")
        
    def extract_image_number(self, file_name):
         # Extrahiere die numerische Bildnummer aus dem Dateinamen
         return int(''.join(filter(str.isdigit, file_name)))
        
    def __len__(self)-> int:
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Laden des Bildes
        img_name = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(img_name)
        
        # Laden der Masken für dieses Bild
        mask_files_for_image = self.mask_files[index * 6 : (index + 1) * 6]
        masks = [Image.open(os.path.join(self.mask_folder, mask_file)) for mask_file in mask_files_for_image]
        
        # Optional: Transformationen anwenden
        if self.transform:
            image = self.transform(image)
            masks = [self.transform(mask) for mask in masks]
            
        # Konvertieren in Tensor und Rückgabe
        image_tensor = torch.tensor(np.array(image))
        masks_tensor = torch.stack([torch.tensor(np.array(mask)) for mask in masks])
    
        return {
            'image': image_tensor,
            'masks': masks_tensor,
        }

transformations = transforms.ToTensor()


asdf = config.IMAGE_DATASET_PATH

dataset = CustomDataset(image_folder=config.IMAGE_DATASET_PATH,
                        mask_folder=config.MASK_DATASET_PATH,
                        transform=transformations)

index=0

example = dataset[index]

# # Überprüfe die Art des zurückgegebenen Objekts
# print("Datentyp von 'example':", type(example))

# # Überprüfe die Schlüssel des zurückgegebenen Dictionaries
# print("Schlüssel des Dictionaries:", example.keys())

# # Überprüfe die Form der Bilddaten (Tensor)
# print("Form des Bild-Tensors:", example['image'].shape)

# # Überprüfe die Form der Maskendaten (Tensor)
# print("Form des Masken-Tensors:", example['masks'].shape)

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
        masks_tensor = example['masks']
        
        image_tensors.append(image_tensor)
        masks_tensors.append(masks_tensor)

    return image_tensors, masks_tensors

# Anwendung der Funktion auf dein CustomDataset
image_tensors, masks_tensors = extract_all_tensors(dataset)

# Ausgabe der Anzahl der extrahierten Bild-Tensoren und Masken-Tensoren
print("Anzahl der extrahierten Bild-Tensoren:", len(image_tensors))
print("Anzahl der extrahierten Masken-Tensoren für jedes Bild:", len(masks_tensors))

print("Erster Tensor: ", image_tensors[0])