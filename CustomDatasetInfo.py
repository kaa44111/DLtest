# %%
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import v2
from torchvision import transforms
import DefineDataset
import config
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch

# %%
# 1. Datensatz und DataLoader erstellen
# Beispiel für die Transformationen-Definition
transform_v2 = v2.Compose([
    v2.ToTensor(),
    #v2.ToDtype(torch.float32, scale=True)
    ])

train_dataset = DefineDataset.CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, transform=transform_v2)
index = 0

example = train_dataset[index]

# Überprüfe die Art des zurückgegebenen Objekts
print("Datentyp von 'example':", type(example))

# Überprüfe die Schlüssel des zurückgegebenen Dictionaries
print("Schlüssel des Dictionaries:", example.keys())

# Überprüfe die Form der Bilddaten (Tensor)
print("Form des Bild-Tensors:", example['image'][0].shape)

# Überprüfe die Form der Maskendaten (Tensor)
print("Form des Masken-Tensors:", example['masks'].shape)

print("############################################################################################ \n")

# %%

train_dataloader = DefineDataset.get_dataloaders()['train']
# Holen des nächsten Batches aus dem DataLoader
batch = next(iter(train_dataloader))

# Extrahieren der Bilder und Masken aus dem Batch
images = batch['image']
masks = batch['masks']

# Drucken der Form der Bilder und Masken
print(f"Form der Bilder DataLoaders: {images[0].shape} -> [Batch-Größe, Farb-Kanäle, Höhe, Breite]")
print(f"Form der Masken DataLoaders: {masks.shape} -> [Batch-Größe, Anzahl der Masken, Höhe, Breite]")

# Berechnung der Gesamtanzahl der Batches
total_batches = len(train_dataloader)

# Ausgabe der Gesamtanzahl der Batches
print("Gesamtanzahl der Batches:", total_batches)


# %%
print("############################################################################################ \n")
# Anwendung der Funktion auf den CustomDataset
image_tensors, masks_tensors = DefineDataset.extract_all_tensors(train_dataset)

# Ausgabe der Anzahl der extrahierten Bild-Tensoren und Masken-Tensoren
print("Anzahl der extrahierten Bild-Tensoren:", len(image_tensors))
print("Anzahl der extrahierten Masken-Tensoren für jedes Bild:", len(masks_tensors))

print("Image Tensor: ", image_tensors[0][0].shape)

masks_tensors_for_image_0 = masks_tensors[index]

# Anzeigen der Form der Maskentensoren für das Bild mit dem Index 0
print("Form der Maskentensoren für das Bild mit Index 0:")
for i, mask_tensor in enumerate(masks_tensors_for_image_0):
    print(f"Maske {i+1}: {mask_tensor.shape}")
# %%
print("____________________________________________________________________________________________________________________")

# Funktion zur Visualisierung von Bild und Masken
def show_image_and_masks(batch):
    
    image_patches = batch['image']  # image_patches hat die Form (batch_size, num_patches, channels, height, width)
    first_image_patches = image_patches[0]  # Extrahiere die ersten Patches des ersten Bildes
    first_six_masks = batch['masks'][0]  # Extrahiere die ersten 6 Masken des ersten Bildes

    # Anzeigen der ersten Patches des Bildes
    plt.figure(figsize=(15, 5))
    num_patches = len(first_image_patches)
    for i in range(num_patches):
        plt.subplot(1, num_patches+ 1, i + 2)
        plt.imshow(first_image_patches[i].permute(1, 2, 0))  # Umwandlung in das erwartete Format (Höhe, Breite, Kanäle)
        plt.title(f'Patch {i + 1}')
        plt.axis('off')
    plt.show()

    # Anzeigen der ersten 6 Masken des Bildes
    plt.figure(figsize=(15, 5))
    for i in range(len(first_six_masks)):
        plt.subplot(1, len(first_six_masks), i + 1)
        plt.imshow(first_six_masks[i], cmap='gray')  # Anzeigen in Graustufen
        plt.title(f'Maske {i + 1}')
        plt.axis('off')
    plt.show()

    # Überprüfen der Anzahl der extrahierten Patches
    print("Anzahl der extrahierten Patches:", len(first_image_patches))

show_image_and_masks(batch=batch)
# %%
