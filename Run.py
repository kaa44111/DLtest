import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import DefineDataset
import config
import UNetModel
import torch.nn as nn
from torch.utils.data import random_split


# 1. Datensatz und DataLoader erstellen
simple_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = DefineDataset.CustomDataset(image_folder=config.IMAGE_DATASET_PATH,
                        mask_folder=config.MASK_DATASET_PATH,
                        transform=simple_transform)

# Anzahl der Bilder für das Trainings- und Validierungsset
train_size = int(config.TEST_SPLIT * len(train_dataset))
val_size = len(train_dataset) - train_size

# Teile den Datensatz in Trainings- und Validierungsset auf
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader für Trainings- und Validierungsset
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)