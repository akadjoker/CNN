import pickle
import numpy as np
import cv2
import os
from tqdm import tqdm

# Carregar dados
train_images = pickle.load(open("full_CNN_train.p", "rb"))
labels = pickle.load(open("full_CNN_labels.p", "rb"))

train_images = np.array(train_images, dtype=np.uint8)  # já estão no intervalo 0-255
labels = np.array(labels, dtype=np.uint8)

# Criar pastas
os.makedirs("images", exist_ok=True)
os.makedirs("masks", exist_ok=True)

# Exportar imagens e masks
for i in tqdm(range(len(train_images)), desc="Exportando"):
    img = train_images[i]
    mask = labels[i].squeeze()  # remover canal extra

    img_filename = f"images/img_{i:05d}.png"
    mask_filename = f"masks/mask_{i:05d}.png"

    cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_filename, mask)

print("Exportação concluída!")

