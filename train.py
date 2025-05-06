import pickle
import numpy as np
from torch.utils.data import DataLoader
import torch
from models import LaneNet
from utils import LaneDataset, train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    # Carregar os dados
    train_images = pickle.load(open("full_CNN_train.p", "rb"))
    labels = pickle.load(open("full_CNN_labels.p", "rb"))

    # Normalizar
    train_images = np.array(train_images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.float32) / 255.0


    # Baralhar
    train_images, labels = shuffle(train_images, labels)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels)

    # Datasets e DataLoaders
    train_dataset = LaneDataset(X_train, y_train)
    val_dataset = LaneDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneNet().to(device)

    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    patience = 10  # early stopping se não melhorar após 10 épocas
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Barra de progresso para treino
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}")

        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Época {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Verificar se é o melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> Novo melhor modelo guardado!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"(Sem melhoria há {epochs_without_improvement} épocas)")

        # Early stopping
        if epochs_without_improvement >= patience:
            print("Early stopping ativado!")
            break

    # Estatísticas finais
    print("\nTreino concluído!")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")

    # Plot
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Curvas de Perda')
    plt.legend()
    plt.grid()
    plt.savefig("training_curves.png")
    plt.show()

if __name__ == '__main__':
    main()

