import cv2
import torch
import numpy as np
import json
import os
from models import LaneNet

# CONFIGURAÇÕES
video_path = "/home/djoker/code/cuda/pt1.mp4"  
frame_width = 160
frame_height = 80
roi_config_file = "roi_config.json"
output_video_path = "output_video.avi"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar modelo treinado
model = LaneNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Abrir vídeo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir vídeo.")
    exit()

# Ler primeiro frame para s tamanho
ret, frame = cap.read()
if not ret:
    print("Erro ao ler primeiro frame.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#//out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width*4, height*4))
out = cv2.VideoWriter('output_segmentado.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_h, frame_w, _ = frame.shape

# Tentar carregar ROI guardado
if os.path.exists(roi_config_file):
    with open(roi_config_file, 'r') as f:
        roi = json.load(f)
else:
    roi = {"top": 50, "bottom": 100, "left": 0, "right": 100}

# Criar janela e sliders
cv2.namedWindow("Frame")

def nothing(x):
    pass

cv2.createTrackbar("Top %", "Frame", roi["top"], 100, nothing)
cv2.createTrackbar("Bottom %", "Frame", roi["bottom"], 100, nothing)
cv2.createTrackbar("Left %", "Frame", roi["left"], 100, nothing)
cv2.createTrackbar("Right %", "Frame", roi["right"], 100, nothing)

print("Ajusta os sliders. Pressiona 's' para salvar ROI ou 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Atualizar ROI dos sliders
    roi["top"] = cv2.getTrackbarPos("Top %", "Frame")
    roi["bottom"] = cv2.getTrackbarPos("Bottom %", "Frame")
    roi["left"] = cv2.getTrackbarPos("Left %", "Frame")
    roi["right"] = cv2.getTrackbarPos("Right %", "Frame")

    # Aplicar ROI
    top_px = int(frame_h * roi["top"] / 100)
    bottom_px = int(frame_h * roi["bottom"] / 100)
    left_px = int(frame_w * roi["left"] / 100)
    right_px = int(frame_w * roi["right"] / 100)

    roi_frame = frame[top_px:bottom_px, left_px:right_px]

    # Se ROI muito pequeno, ignorar este frame
    if roi_frame.shape[0] < 10 or roi_frame.shape[1] < 10:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # Redimensionar ROI para input do modelo
    roi_resized = cv2.resize(roi_frame, (frame_width, frame_height))
    frame_input = roi_resized.astype(np.float32) / 255.0
    frame_input = torch.tensor(frame_input).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inferência
    with torch.no_grad():
        mask = model(frame_input).cpu().squeeze().numpy()

    # Limpeza da máscara
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    # Redimensionar máscara para tamanho do ROI original
    mask_up = cv2.resize(mask_clean, (right_px - left_px, bottom_px - top_px))

    # Criar máscara colorida
    mask_color = np.zeros_like(roi_frame)
    mask_color[:, :, 1] = mask_up  # verde

    # Aplicar máscara sobre ROI
    overlay = cv2.addWeighted(roi_frame, 1.0, mask_color, 0.5, 0)

    # Inserir o overlay de volta no frame original
    frame_overlay = frame.copy()
    frame_overlay[top_px:bottom_px, left_px:right_px] = overlay

    # Mostrar retângulo do ROI
    cv2.rectangle(frame_overlay, (left_px, top_px), (right_px, bottom_px), (0, 255, 0), 2)

    cv2.imshow("Frame", frame_overlay)
    out.write(frame_overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        with open(roi_config_file, 'w') as f:
            json.dump(roi, f, indent=4)
        print("ROI save", roi_config_file)

out.release()
cap.release()
cv2.destroyAllWindows()

