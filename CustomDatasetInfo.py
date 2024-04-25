import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import DefineDataset
import config
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# 1. Datensatz und DataLoader erstellen
transform = Compose([ToTensor()])  
train_dataset = DefineDataset.CustomDataset(image_folder=config.IMAGE_DATASET_PATH,
                        mask_folder=config.MASK_DATASET_PATH,
                        transform=transform)
index=0

example = train_dataset[index]

# Überprüfe die Art des zurückgegebenen Objekts
print("Datentyp von 'example':", type(example))

# Überprüfe die Schlüssel des zurückgegebenen Dictionaries
print("Schlüssel des Dictionaries:", example.keys())

# Überprüfe die Form der Bilddaten (Tensor)
print("Form des Bild-Tensors:", example['image'].shape)

# Überprüfe die Form der Maskendaten (Tensor)
print("Form des Masken-Tensors:", example['masks'].shape)

print("############################################################################################ \n")


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
print(f"Form der Bilder DataLoaders: {images.shape} -> [Batch-Größe, Farb-Kanäle, Höhe, Breite]")
print(f"Form der Masken DataLoaders: {masks.shape} -> [Batch-Größe, Anzahl der Masken, Höhe, Breite]")

# Berechnung der Gesamtanzahl der Batches
total_batches = len(train_loader)

# Ausgabe der Gesamtanzahl der Batches
print("Gesamtanzahl der Batches:", total_batches)

print("############################################################################################ \n")
# Anwendung der Funktion auf den CustomDataset
image_tensors, masks_tensors = DefineDataset.extract_all_tensors(train_dataset)

# Ausgabe der Anzahl der extrahierten Bild-Tensoren und Masken-Tensoren
print("Anzahl der extrahierten Bild-Tensoren:", len(image_tensors))
print("Anzahl der extrahierten Masken-Tensoren für jedes Bild:", len(masks_tensors))

print("Image Tensor: ", image_tensors[0].shape)

masks_tensors_for_image_0 = masks_tensors[index]

# Anzeigen der Form der Maskentensoren für das Bild mit dem Index 0
print("Form der Maskentensoren für das Bild mit Index 0:")
for i, mask_tensor in enumerate(masks_tensors_for_image_0):
    print(f"Maske {i+1}: {mask_tensor.shape}")

print("____________________________________________________________________________________________________________________")

# Funktion zur Visualisierung von Bild und Masken
def show_image_and_masks(image, masks):
    plt.figure(figsize=(15, 5))
    
    # Bild anzeigen
    plt.subplot(1, 7, 1)
    plt.imshow(image.permute(1, 2, 0))  # Permutiere die Dimensionen für die Anzeige
    plt.title('Bild')
    plt.axis('off')
    
    # Masken anzeigen
    for i, mask in enumerate(masks):
        plt.subplot(1, 7, i+2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Maske {i+1}')
        plt.axis('off')
    
    plt.show()

# Lade ein Beispiel aus dem DataLoader
for data in train_loader:
    image = data['image'][0]  # Erstes Bild im Batch
    masks = data['masks'][0]  # Masken zum ersten Bild
    show_image_and_masks(image, masks)
    break  # Nur das erste Beispiel anzeigen