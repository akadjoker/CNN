import pygame
import sys
import os
import numpy as np
from PIL import Image
import json

class LabelTool:
    def __init__(self):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Ferramenta de Anotação de Imagens')
        
        # Definir cores
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        
        # Configurações de fonte
        self.font = pygame.font.SysFont('Arial', 20)
        
        # Lista de imagens e índice atual
        self.images_dir = "images"  # Pasta com as imagens
        self.labels_dir = "labels"  # Pasta para salvar as labels
        self.image_files = []
        self.current_index = 0
        self.load_image_list()
        
        # Pontos e estados
        self.points = []
        self.dragging_point = None
        self.dragging_index = -1
        self.show_mask = False
        self.mask_surface = None
        
        # Criar pasta labels se não existir
        if not os.path.exists(self.labels_dir):
            os.makedirs(self.labels_dir)
            
        # Carregar primeira imagem
        self.load_current_image()
        
    def load_image_list(self):
        """Carregar a lista de imagens da pasta"""
        if os.path.exists(self.images_dir):
            self.image_files = [f for f in os.listdir(self.images_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            self.image_files.sort()
            print(f"Encontradas {len(self.image_files)} imagens")
        else:
            print(f"Diretório {self.images_dir} não encontrado. Criando...")
            os.makedirs(self.images_dir)
            
    def load_current_image(self):
        """Carregar a imagem atual e seus pontos se existirem"""
        if not self.image_files:
            self.current_image = None
            self.image_rect = None
            return
            
        image_path = os.path.join(self.images_dir, self.image_files[self.current_index])
        self.points = []
        self.show_mask = False
        self.mask_surface = None
        
        try:
            # Carregar imagem
            self.current_image = pygame.image.load(image_path).convert_alpha()
            
            # Redimensionar mantendo proporção
            img_width, img_height = self.current_image.get_size()
            scale = min(
                (self.screen_width - 40) / img_width,
                (self.screen_height - 80) / img_height
            )
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            self.current_image = pygame.transform.scale(self.current_image, (new_width, new_height))
            self.image_rect = self.current_image.get_rect(center=(self.screen_width//2, self.screen_height//2))
            
            # Carregar pontos se existirem
            self.load_points()
            
            pygame.display.set_caption(f'Imagem: {self.image_files[self.current_index]} ({self.current_index+1}/{len(self.image_files)})')
            
        except Exception as e:
            print(f"Erro ao carregar imagem {image_path}: {e}")
            self.current_image = None
            self.image_rect = None
            
    def load_points(self):
        """Carregar pontos existentes do arquivo de label"""
        if not self.image_files:
            return
            
        base_name = os.path.splitext(self.image_files[self.current_index])[0]
        label_file = os.path.join(self.labels_dir, f"{base_name}.json")
        
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as f:
                    data = json.load(f)
                    # Converter pontos de coordenadas originais para coordenadas da tela
                    if 'points' in data:
                        img_width, img_height = data.get('original_size', (1, 1))
                        scale_x = self.image_rect.width / img_width
                        scale_y = self.image_rect.height / img_height
                        
                        self.points = []
                        for x, y in data['points']:
                            screen_x = self.image_rect.left + int(x * scale_x)
                            screen_y = self.image_rect.top + int(y * scale_y)
                            self.points.append((screen_x, screen_y))
                            
                        print(f"Carregados {len(self.points)} pontos do arquivo {label_file}")
            except Exception as e:
                print(f"Erro ao carregar pontos: {e}")
        
    def save_points(self):
        """Salvar pontos no arquivo de label"""
        if not self.image_files or not self.current_image:
            return
            
        base_name = os.path.splitext(self.image_files[self.current_index])[0]
        label_file = os.path.join(self.labels_dir, f"{base_name}.json")
        
        # Converter pontos de coordenadas da tela para coordenadas originais
        original_points = []
        if self.points:
            img_width, img_height = self.current_image.get_size()
            
            for screen_x, screen_y in self.points:
                # Converter para coordenadas relativas à imagem
                img_x = screen_x - self.image_rect.left
                img_y = screen_y - self.image_rect.top
                
                # Normalizar para coordenadas originais
                original_points.append((img_x, img_y))
        
        data = {
            'filename': self.image_files[self.current_index],
            'original_size': self.current_image.get_size(),
            'points': original_points
        }
        
        try:
            with open(label_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Pontos salvos em {label_file}")
            
            # Salvar máscara se houver pontos
            #if len(self.points) >= 3:
            #    self.save_mask()
                
        except Exception as e:
            print(f"Erro ao salvar pontos: {e}")
    
    def save_mask(self):
        """Salvar máscara como imagem PNG"""
        if not self.image_files or len(self.points) < 3:
            return
            
        base_name = os.path.splitext(self.image_files[self.current_index])[0]
        mask_file = os.path.join(self.labels_dir, f"{base_name}_mask.png")
        
        try:
            # Criar superfície para a máscara
            surface = pygame.Surface(self.current_image.get_size(), pygame.SRCALPHA)
            
            # Converter pontos para coordenadas relativas à imagem
            img_points = []
            for screen_x, screen_y in self.points:
                img_x = screen_x - self.image_rect.left
                img_y = screen_y - self.image_rect.top
                img_points.append((img_x, img_y))
            
            # Desenhar polígono preenchido
            if len(img_points) >= 3:
                pygame.draw.polygon(surface, (255, 255, 255, 128), img_points)
            
            # Salvar superfície como PNG
            pygame.image.save(surface, mask_file)
            print(f"Máscara salva em {mask_file}")
            
        except Exception as e:
            print(f"Erro ao salvar máscara: {e}")
    
    def generate_mask_preview(self):
        """Gerar prévia da máscara"""
        if not self.current_image or len(self.points) < 3:
            self.mask_surface = None
            return
            
        # Criar superfície para a máscara
        self.mask_surface = pygame.Surface(self.current_image.get_size(), pygame.SRCALPHA)
        
        # Converter pontos para coordenadas relativas à imagem
        img_points = []
        for screen_x, screen_y in self.points:
            img_x = screen_x - self.image_rect.left
            img_y = screen_y - self.image_rect.top
            img_points.append((img_x, img_y))
        
        # Desenhar polígono preenchido
        if len(img_points) >= 3:
            pygame.draw.polygon(self.mask_surface, (255, 0, 0, 128), img_points)
    
    def next_image(self):
        """Carregar próxima imagem"""
        if not self.image_files:
            return
            
        self.save_points()
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_current_image()
    
    def prev_image(self):
        """Carregar imagem anterior"""
        if not self.image_files:
            return
            
        self.save_points()
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_current_image()
    
    def add_point(self, pos):
        """Adicionar ponto na posição clicada"""
        if self.image_rect and self.image_rect.collidepoint(pos):
            self.points.append(pos)
            self.show_mask = False
    
    def find_closest_point(self, pos, max_dist=10):
        """Encontrar o ponto mais próximo dentro de uma distância máxima"""
        if not self.points:
            return -1
            
        distances = [((p[0] - pos[0])**2 + (p[1] - pos[1])**2)**0.5 for p in self.points]
        min_idx = distances.index(min(distances))
        
        if distances[min_idx] <= max_dist:
            return min_idx
        return -1
    
    def run(self):
        """Loop principal"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_points()
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.save_points()
                        running = False
                    elif event.key == pygame.K_RIGHT:
                        self.next_image()
                    elif event.key == pygame.K_LEFT:
                        self.prev_image()
                    elif event.key == pygame.K_f:
                        # Gerar máscara com F
                        self.show_mask = True
                        self.generate_mask_preview()
                    elif event.key == pygame.K_s:
                        # Salvar com S
                        self.save_points()
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        # Remover último ponto
                        if self.points:
                            self.points.pop()
                            self.show_mask = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Botão esquerdo
                        # Verificar se clicou em um ponto existente
                        point_idx = self.find_closest_point(event.pos)
                        if point_idx >= 0:
                            self.dragging_point = event.pos
                            self.dragging_index = point_idx
                        elif self.image_rect and self.image_rect.collidepoint(event.pos):
                            # Adicionar novo ponto
                            self.add_point(event.pos)
                            self.show_mask = False
                    
                    elif event.button == 3:  # Botão direito
                        # Adicionar ponto com botão direito
                        if self.image_rect and self.image_rect.collidepoint(event.pos):
                            self.add_point(event.pos)
                            self.show_mask = False
                    
                    elif event.button == 4:  # Roda para cima
                        self.prev_image()
                    
                    elif event.button == 5:  # Roda para baixo
                        self.next_image()
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.dragging_index >= 0:
                        self.dragging_point = None
                        self.dragging_index = -1
                        self.show_mask = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_index >= 0 and self.dragging_point:
                        # Mover ponto arrastado
                        if self.image_rect and self.image_rect.collidepoint(event.pos):
                            self.points[self.dragging_index] = event.pos
                            self.show_mask = False
            
            # Limpar tela
            self.screen.fill(self.BLACK)
            
            # Desenhar imagem
            if self.current_image and self.image_rect:
                self.screen.blit(self.current_image, self.image_rect)
                
                # Desenhar máscara se ativada
                if self.show_mask and self.mask_surface:
                    self.screen.blit(self.mask_surface, self.image_rect)
                
                # Desenhar pontos
                for i, point in enumerate(self.points):
                    pygame.draw.circle(self.screen, self.RED, point, 5)
                    point_label = self.font.render(str(i), True, self.YELLOW)
                    self.screen.blit(point_label, (point[0] + 8, point[1] - 8))
                
                # Desenhar linhas entre pontos
                if len(self.points) > 1:
                    pygame.draw.lines(self.screen, self.GREEN, True, self.points, 2)
            
            # Desenhar instruções
            instructions = [
                "Botão Esquerdo: Adicionar/Mover pontos",
                "Botão Direito: Adicionar pontos",
                "F: Visualizar máscara",
                "S: Salvar pontos e máscara",
                "Setas/Roda: Navegar entre imagens",
                "Delete/Backspace: Remover último ponto",
                "ESC: Sair"
            ]
            
            y_pos = 10
            for instruction in instructions:
                text = self.font.render(instruction, True, self.WHITE)
                self.screen.blit(text, (10, y_pos))
                y_pos += 25
            
            # Desenhar informações da imagem atual
            if self.image_files:
                info_text = f"Imagem: {self.current_index+1}/{len(self.image_files)} - {self.image_files[self.current_index]}"
                text = self.font.render(info_text, True, self.WHITE)
                self.screen.blit(text, (10, self.screen_height - 30))
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = LabelTool()
    app.run()
