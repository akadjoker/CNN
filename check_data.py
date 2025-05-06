import pickle
import numpy as np
import matplotlib.pyplot as plt

# Carregar os ficheiros .p
train_images = pickle.load(open("full_CNN_train.p", "rb"))
labels = pickle.load(open("full_CNN_labels.p", "rb"))

# Converter para NumPy e normalizar
train_images = np.array(train_images, dtype=np.float32) / 255.0
labels = np.array(labels, dtype=np.float32)

print("Formato das imagens:", train_images.shape)
print("Formato dos labels:", labels.shape)

print("Valor mínimo dos labels:", labels.min())
print("Valor máximo dos labels:", labels.max())


print("Valor mínimo dos images:", train_images.min())
print("Valor máximo dos images:", train_images.max())


# Mostrar uma imagem e o seu label correspondente
idx = 0  # podes mudar este índice para ver outras imagens
image = train_images[idx]
label = labels[idx]

# Se a imagem tiver canais, mostrar como RGB, senão grayscale
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
if image.ndim == 3 and image.shape[2] == 3:
    plt.imshow(image)
else:
    plt.imshow(image.squeeze(), cmap='gray')
plt.title("Imagem")

plt.subplot(1, 2, 2)
plt.imshow(label.squeeze(), cmap='gray')
plt.title("Label")
plt.savefig("data.png")
plt.show()

