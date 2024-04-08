import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import config

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        
        # Liste der Dateinamen in den Ordnern und sortieren nach Bildnummer
        self.image_files = sorted(os.listdir(image_folder), key=self.extract_image_number)
        self.mask_files = sorted(os.listdir(mask_folder), key=self.extract_image_number)
        
        # Überprüfen, ob die Anzahl der Dateien übereinstimmt
        if len(self.image_files) != len(self.mask_files) / 6:
            raise ValueError("Die Anzahl der Bilder und Masken stimmt nicht überein.")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Laden des Bildes
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name)
        
        # Laden der Masken für dieses Bild
        mask_files_for_image = self.mask_files[idx*6:(idx+1)*6]
        masks = [Image.open(os.path.join(self.mask_folder, mask_file)) for mask_file in mask_files_for_image]
        
        # Optional: Transformationen anwenden
        if self.transform:
            image = self.transform(image)
            masks = [self.transform(mask) for mask in masks]
        
        # Konvertieren in Tensor und Rückgabe
        return {'image': torch.tensor(np.array(image)).permute(2, 0, 1),
                'masks': torch.stack([torch.tensor(np.array(mask)) for mask in masks])}

    def extract_image_number(self, file_name):
        return int(file_name.split('.')[0])
    

# dataset = CustomDataset(config.IMAGE_DATASET_PATH,config.MASK_DATASET_PATH)