import cv2
import torch
import numpy as np
from models import LaneNet

# --- CONFIGURAÇÕES ---
video_path = "highway.avi"  
output_video_path = "output_video.mp4"
frame_width = 160
frame_height = 80

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
model = LaneNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

 
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir vídeo.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width*4, frame_height*4))  # upscale para melhor visualização

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    frame_input = frame_resized.astype(np.float32) / 255.0
    frame_input = torch.tensor(frame_input).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(frame_input).cpu().squeeze().numpy()

    # Limpeza de ruído
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    mask_color = np.zeros_like(frame_resized)
    mask_color[:, :, 1] = mask_clean  # verde

    overlay = cv2.addWeighted(frame_resized, 0.8, mask_color, 0.9, 0)
    overlay_up = cv2.resize(overlay, (frame_width*4, frame_height*4))

    cv2.imshow('Frame ', overlay_up)
    out.write(overlay_up)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

 

