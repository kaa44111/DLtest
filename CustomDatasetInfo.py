import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import DefineDataset
import config
from torch.utils.data import random_split

# 1. Datensatz und DataLoader erstellen
transform = Compose([ToTensor()])  
train_dataset = DefineDataset.CustomDataset(image_folder=config.IMAGE_DATASET_PATH,
                        mask_folder=config.MASK_DATASET_PATH,
                        transform=transform)

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

# Holen des nächsten Batches aus dem DataLoader
batch = next(iter(train_loader))

# Extrahieren der Bilder und Masken aus dem Batch
images = batch['image']
masks = batch['masks']

# Drucken der Form der Bilder und Masken
print(f"Form der Bilder: {images.shape} -> [Batch-Größe, Kanäle, Höhe, Breite]")
print(f"Form der Masken: {masks.shape} -> [Batch-Größe, Anzahl der Masken, Höhe, Breite]")

# Berechnung der Gesamtanzahl der Batches
total_batches = len(train_loader)

# Ausgabe der Gesamtanzahl der Batches
print("Gesamtanzahl der Batches:", total_batches)