import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm

# Caminhos das pastas
images_dir = "out/images"
masks_dir = "out/mask"

# Definir tamanho alvo (para bater certo com o modelo)
target_size = (160, 80)  # largura, altura

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

assert len(image_files) == len(mask_files), "Número de imagens e máscaras não corresponde!"

train_images = []
labels = []

for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processando"):

    # Ler imagem e máscara
    img = cv2.imread(os.path.join(images_dir, img_file))
    mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)

    # Redimensionar
    img_resized = cv2.resize(img, target_size)
    mask_resized = cv2.resize(mask, target_size)

    # Converter máscara para 1 canal e garantir tipo uint8
    mask_resized = np.expand_dims(mask_resized, axis=-1).astype(np.uint8)

    # Converter imagem para RGB
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    train_images.append(img_resized)
    labels.append(mask_resized)

train_images = np.array(train_images, dtype=np.uint8)
labels = np.array(labels, dtype=np.uint8)

print("Shapes finais:")
print("train_images:", train_images.shape)
print("labels:", labels.shape)

# Guardar nos ficheiros .p
with open("images.p", "wb") as f:
    pickle.dump(train_images, f)

with open("labels.p", "wb") as f:
    pickle.dump(labels, f)

print("Ficheiros  train.p e  labels.p criados com sucesso!")

