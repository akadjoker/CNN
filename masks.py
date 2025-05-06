import os
import cv2
import torch
import numpy as np
from models import LaneNet
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
input_folder = "/home/djoker/code/cuda/LandDetection/images"
output_folder = "output_masks"

frame_width = 160
frame_height = 80

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Criar pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Carregar modelo treinado
model = LaneNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Listar imagens
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in tqdm(image_files, desc="Gerando máscaras"):
    # Ler imagem
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path)

    # Redimensionar e normalizar
    frame_resized = cv2.resize(img, (frame_width, frame_height))
    frame_input = frame_resized.astype(np.float32) / 255.0
    frame_input = torch.tensor(frame_input).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inferência
    with torch.no_grad():
        mask = model(frame_input).cpu().squeeze().numpy()

    # Limpeza de ruído
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    # Salvar máscara
    mask_filename = os.path.splitext(img_file)[0] + "_mask.png"
    cv2.imwrite(os.path.join(output_folder, mask_filename), mask_clean)

print("Máscaras geradas em:", output_folder)

