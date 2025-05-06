import os
import cv2
import numpy as np

IMAGE_DIR = "out/images"
MASK_DIR = "out/mask"

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])

if not image_files:
    print("Nenhuma imagem encontrada em", IMAGE_DIR)
    exit()

index = 0
scale = 4  # Escala de visualização

while True:
    image_name = image_files[index]
    image_path = os.path.join(IMAGE_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, image_name)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Erro ao carregar {image_name}")
        continue

    # Criar máscara colorida (verde)
    mask_color = cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)])
    overlay = cv2.addWeighted(image, 1.0, mask_color, 0.5, 0)

    # Escalar imagem
    overlay_scaled = cv2.resize(overlay, (image.shape[1]*scale, image.shape[0]*scale))

    cv2.imshow("Preview", overlay_scaled)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == 83 or key == ord('d'):  # seta direita ou D
        index = (index + 1) % len(image_files)
    elif key == 81 or key == ord('a'):  # seta esquerda ou A
        index = (index - 1) % len(image_files)

cv2.destroyAllWindows()

