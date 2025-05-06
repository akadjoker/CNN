import pygame
import os
import json
import glob

# --- CONFIGURAÇÕES ---
IMAGE_FOLDER = "images"
SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 720

POINT_RADIUS = 5
POINT_COLOR = (0, 255, 0)  # verde
FILL_COLOR = (0, 255, 0, 100)  # verde transparente

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ferramenta de Anotação (Label Tool)")

font = pygame.font.SysFont(None, 24)

# --- Funções ---

def load_image_paths():
    return sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) + 
                  glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))

def resize_image(image, target_width, target_height):
    return pygame.transform.scale(image, (target_width, target_height))

def load_points(txt_path):
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return [tuple(map(int, line.strip().split())) for line in f]
    return []

def save_points(txt_path, points):
    with open(txt_path, "w") as f:
        for p in points:
            f.write(f"{p[0]} {p[1]}\n")

def draw_points(surface, points):
    for p in points:
        pygame.draw.circle(surface, POINT_COLOR, p, POINT_RADIUS)

def fill_polygon(surface, points):
    if len(points) >= 3:
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, FILL_COLOR, points)
        surface.blit(s, (0,0))

def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

# --- Main ---

image_paths = load_image_paths()
if not image_paths:
    print("Nenhuma imagem encontrada na pasta", IMAGE_FOLDER)
    pygame.quit()
    exit()

index = 0
points = []

running = True
fill = False

while running:
    img_path = image_paths[index]
    img_name = os.path.basename(img_path)
    img = pygame.image.load(img_path)
    img_resized = resize_image(img, SCREEN_WIDTH, SCREEN_HEIGHT)

    # Carregar pontos se existirem
    txt_path = os.path.splitext(img_path)[0] + "_lines.txt"
    points = load_points(txt_path)

    moving_point = None

    while True:
        screen.fill((0, 0, 0))
        screen.blit(img_resized, (0, 0))

        if fill:
            fill_polygon(screen, points)

        draw_points(screen, points)

        label = font.render(f"Imagem: {img_name} | Teclas: → próximo, ← anterior, s salvar, q sair, f fill", True, (255,255,255))
        screen.blit(label, (10, 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                if event.button == 1:  # Botão esquerdo → mover ponto
                    for i, p in enumerate(points):
                        if distance(p, (x, y)) < 10:
                            moving_point = i
                            break

                elif event.button == 3:  # Botão direito → adicionar ponto
                    points.append((x, y))

            elif event.type == pygame.MOUSEBUTTONUP:
                moving_point = None

            elif event.type == pygame.MOUSEMOTION and moving_point is not None:
                points[moving_point] = event.pos

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    save_points(txt_path, points)
                    index = (index + 1) % len(image_paths)
                    break  # sair para carregar nova imagem

                elif event.key == pygame.K_LEFT:
                    save_points(txt_path, points)
                    index = (index - 1) % len(image_paths)
                    break

                elif event.key == pygame.K_s:
                    save_points(txt_path, points)
                    print(f"Pontos salvos para {img_name}")

                elif event.key == pygame.K_f:
                    fill = not fill

                elif event.key == pygame.K_q:
                    running = False
                    break

    if not running:
        break

pygame.quit()

