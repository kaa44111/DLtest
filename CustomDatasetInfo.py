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
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
])

# first_image,first_masks = train_dataset[0] # Index 0 für das erste Bild im Datensatz

train_dataset = DefineDataset.CustomDataset(config.IMAGE_DATASET_PATH, config.MASK_DATASET_PATH, transform=transform_v2,patch_size=64)
index = 0

first_image,first_masks = train_dataset[index]

# Überprüfe die Art des zurückgegebenen Objekts
print("Datentyp:", type(first_image))

# Überprüfe die Form der Bilddaten (Tensor)
print("Form des Bild-Tensors:", first_image.shape)

# Überprüfe die Form der Bilddaten (Tensor)
print("Form des Bild Patches-Tensors:", first_image[0].shape)

# Überprüfe die Form der Maskendaten (Tensor)
print("Form des Masken-Tensors:", first_masks.shape)

print("############################################################################################ \n")

# %%

data_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
# Holen des nächsten Batches aus dem DataLoader
batch = next(iter(data_loader))

# Extrahieren der Bilder und Masken aus dem Batch
images, masks = batch

# Drucken der Form der Bilder und Masken
print(f"Form der Bilder DataLoaders: {images.shape} -> [Batch-Größe, Anzahl der Patches, Farb-Kanäle, Höhe, Breite]")
print(f"Form der Masken DataLoaders: {masks.shape} -> [Batch-Größe, Anzahl der Masken, Höhe, Breite]")

# Berechnung der Gesamtanzahl der Batches
total_batches = len(data_loader)

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
# # %%
# print("____________________________________________________________________________________________________________________")

# # Funktion zur Visualisierung von Bild und Masken
# def show_image_and_masks(batch):
    
#     image_patches, first_six_masks = batch[0]  # Extrahiere die ersten Patches des ersten Bildes

#     num_patches = len(image_patches)

#     plt.figure(figsize=(15, 5))
    
#     for i in range(num_patches):
#         plt.subplot(1, num_patches + 1, i + 2)
#         plt.imshow(image_patches[i].permute(1, 2, 0))  # Umwandlung in das erwartete Format (Höhe, Breite, Kanäle)
#         plt.title(f'Bild Patch {i+1}')
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()

#     # Anzeigen der ersten 6 Masken des Bildes
#     plt.figure(figsize=(15, 5))
#     for i in range(len(first_six_masks)):
#         plt.subplot(1, len(first_six_masks), i + 1)
#         plt.imshow(first_six_masks[i], cmap='gray')  # Anzeigen in Graustufen
#         plt.title(f'Maske {i + 1}')
#         plt.axis('off')
#     plt.show()

#     # Überprüfen der Anzahl der extrahierten Patches
#     print("Anzahl der extrahierten Patches:", image_patches.shape)

# show_image_and_masks(batch=batch)
# # %%
